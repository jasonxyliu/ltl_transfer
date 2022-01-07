class CurriculumLearner:
    """
    Decides when to stop one task and which to execute next
    In addition, it controls how many steps the agent has given so far
    """
    def __init__(self, tasks, r_good=0.9, num_steps=100, min_steps=1000, total_steps=100000):
        """Parameters
        -------
        tasks: list of strings
            list with the path to the ltl sketch for each task
        r_good: float
            suceess rate threshold to decide moving to the next task
        num_steps: int
            max number of steps that the agent has to complete the task.
            if it does it, we consider a hit on its 'suceess rate'
            (this emulates considering the average reward after running a rollout for 'num_steps')
        min_steps: int
            minimum number of training steps required to the agent before considering moving to another task
        total_steps: int
            total number of training steps that the agent has to learn all the tasks
        """
        self.tasks = tasks
        self.r_good = r_good
        self.num_steps = num_steps
        self.min_steps = min_steps
        self.total_steps = total_steps

    def restart(self):
        self.current_step = 0
        self.succ_rate = {}
        for t in self.tasks:
            self.succ_rate[t] = (0, 0)  # (hits, total)
        self.current_task = -1

    def add_step(self):
        self.current_step += 1

    def get_current_step(self):
        return self.current_step

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def get_next_task(self):
        self.last_restart = -1
        self.current_task = (self.current_task+1) % len(self.tasks)
        return self.get_current_task()

    def get_current_task(self):
        return self.tasks[self.current_task]

    def update_succ_rate(self, step, reward):
        t = self.get_current_task()
        hits, total = self.succ_rate[t]
        if reward == 1 and (step-self.last_restart) <= self.num_steps:
            hits += 1.0
        total += 1.0
        self.succ_rate[t] = (hits, total)
        self.last_restart = step

    def stop_task(self, step):
        return self.min_steps <= step and self.r_good < self.get_succ_rate()

    def get_succ_rate(self):
        t = self.get_current_task()
        hits, total = self.succ_rate[t]
        return 0 if total == 0 else (hits/total)

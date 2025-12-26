import math

import pyglet
from pyglet.window import key


class ManualControl:
    def __init__(self, env, no_time_limit: bool, domain_rand: bool):
        self.env = env.unwrapped

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

    def run(self):
        print("============")
        print("Instructions")
        print("============")
        print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
        print("============")

        self.env.reset()

        # Create the display window
        self.env.render()

        env = self.env

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation
            """

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.env.reset()
                self.env.render()
                return

            if symbol == key.ESCAPE:
                self.env.close()

            if symbol == key.UP:
                self.step("LookUp")
            elif symbol == key.DOWN:
                self.step("LookDown")
            elif symbol == key.LEFT:
                self.step("RotateLeft")
            elif symbol == key.RIGHT:
                self.step("RotateRight")

            elif symbol == key.Z:
                self.step("MoveAhead")
            elif symbol == key.S:
                self.step("MoveBack")
            elif symbol == key.Q:
                self.step("MoveLeft")
            elif symbol == key.D:
                self.step("MoveRight")
            else:
                print("unknown key: %d" % symbol)

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            pass

        @env.unwrapped.window.event
        def on_draw():
            self.env.render()

        @env.unwrapped.window.event
        def on_close():
            pyglet.app.exit()

        # Enter main event loop
        pyglet.app.run()

        self.env.close()

    def step(self, action):
        print(
            f"step {self.env.unwrapped.step_count + 1}/{self.env.unwrapped.max_episode_steps}: {action}"
        )

        obs, reward, termination, truncation, info = self.env.step(action)

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            self.env.reset()

        self.env.render()


class AutoControl(ManualControl):
    def __init__(self, agent, sleep: float = 0.2, **kwargs):
        self.agent = agent
        self.sleep = sleep
        super().__init__(env=agent.env, **kwargs)

    def run(self):
        import pyglet

        o, info = self.env.reset()
        self.agent.update(info["state_value"])
        self.env.render()

        def update(dt):
            action_str = self.agent.act()
            o, r, termination, truncation, info = self.env.step(action_str)
            self.agent.update(info["state_value"])
            self.env.render()

        # Schedule agent actions
        pyglet.clock.schedule_interval(update, self.sleep)

        # Start pyglet event loop (must be last)
        pyglet.app.run()

        self.env.close()


if __name__ == "__main__":
    from env import ManyObjectsEnv
    from agent import BOAgent, RFModel

    env = ManyObjectsEnv(n=20, grid_size=16, render_mode="agent")

    agent = BOAgent(
        surrogate_model=RFModel(),
        input_space=[pose.discretize() for pose in env.enumerate_poses()],
        env=env,
    )

    c = AutoControl(agent=agent, no_time_limit=True, domain_rand=False, sleep=0.5)

    c.run()

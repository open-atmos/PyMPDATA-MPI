class _Scenario:
    pass

    def advance(self, dataset, output_steps, x_range):
        steps_done = 0
        for i1, output_step in enumerate(output_steps):
            n_steps = output_step - steps_done
            self.solver.advance(n_steps=n_steps)
            steps_done += n_steps
            dataset[x_range, :, i1] = self.solver.advectee.get()

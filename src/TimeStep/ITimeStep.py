from pandas import DataFrame


class ITimeStep:
    def define_time_step(self, hits: DataFrame):
        """
        Define the pseudo time step for the given event data.

        :param DataFrame hits: The hit data to define the pseudo time step for
        """

        raise NotImplementedError

    def get_num_time_steps(self) -> int:
        """
        Get the number of pseudo time steps.

        :return int: The number of pseudo time steps
        """

        raise NotImplementedError

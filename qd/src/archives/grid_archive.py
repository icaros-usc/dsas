"""Custom GridArchive."""
import gin
import ribs.archives


@gin.configurable
class GridArchive(ribs.archives.GridArchive):
    """Based on pyribs GridArchive.

    This archive records history of its objectives and behavior values if
    record_history is True. Before each generation, call new_history_gen() to
    start recording history for that gen. new_history_gen() must be called
    before calling add() for the first time.
    """

    def __init__(self, *args, record_history=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._record_history = record_history
        self._history = [] if self._record_history else None

    def new_history_gen(self):
        """Starts a new generation in the history."""
        if self._record_history:
            self._history.append([])

    def history(self):
        """Gets the current history."""
        return self._history

    def add_single(self, solution, objective, measures, metadata=None):

        status, value = super().add_single(
            solution,
            objective,
            measures,
            metadata,
        )

        # Save objective and measures in history, even if not inserted into the
        # archive.
        if self._record_history:
            self._history[-1].append(["add_single", objective, measures])

        return status, value

    def add(self,
            solution_batch,
            objective_batch,
            measures_batch,
            metadata_batch=None):

        status_batch, value_batch = super().add(
            solution_batch,
            objective_batch,
            measures_batch,
            metadata_batch,
        )

        if self._record_history:
            self._history[-1].append(["add", objective_batch, measures_batch])

        return status_batch, value_batch

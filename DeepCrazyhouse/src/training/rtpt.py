"""
@file: rtpt.py
Created on 14.08.20
@project: CrazyAra
@author: Quentin Delfosse

RTPT class to rename your processes giving information on who is launching the
process, and the remaining time for it.
Created to be used with our AIML IRON table.

Example usage:
```python
rtpt = RTPT(name_initials='KK', base_title='ScriptName', number_of_epochs=num_epochs, epoch_n=0)
for epoch in range(num_epochs):
  rtpt.epoch_starts()
  train()
  test()
  rtpt.epoch_ends()
```
"""
import datetime
from setproctitle import setproctitle


class RTPT():
    """
    RemainingTimeToProcessTitle
    """
    def __init__(self, name_initials, base_title, number_of_epochs, epoch_n=0):
        """
        Initialize the RTPT object
        !!! PLEASE GIVE A SHORT PROCESS TITLE
            (otherwise you won't be able to get the time)
        !!! PLEASE DO NOT USE SPACE (use underscore _ instead)

        parameters:
         * name_initials: QD for Quentin Delfosse
         * base_title: The title you want your process to have
        The number of epochs you have in your experiment

        """
        assert len(base_title) < 30
        self.base_title = "@" + name_initials + "_" + base_title + "#"
        self._last_epoch_start = None
        self._epoch_n = epoch_n
        self._number_of_epochs = number_of_epochs
        setproctitle(self.base_title + "first_epoch")

    def epoch_starts(self):
        """
        To be called at the start of the epoch
        """
        self._last_epoch_start = datetime.datetime.now()
        self._epoch_n += 1

    def epoch_ends(self):
        """
        To be called at the end of the epoch
        """
        last_epoch_duration = datetime.datetime.now() - self._last_epoch_start
        remaining_epochs = self._number_of_epochs - self._epoch_n
        remaining_time = str(last_epoch_duration * remaining_epochs).split(".")[0]
        if "day" in remaining_time:
            days = remaining_time.split(" day")[0]
            rest = remaining_time.split(", ")[1]
        else:
            days = 0
            rest = remaining_time
        complete_title = self.base_title + f"{days}d:{rest}"
        setproctitle(complete_title)

"""Read the PHREEQC run files and keep them in one cached RunTable.

This package only prepares the data. It reads the run files of each rock,
checks them, puts all runs in one table, and saves the table to a cache.
The optimization code uses the table. It does not read the run files.
"""

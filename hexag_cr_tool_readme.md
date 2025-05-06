### Tue May  6 23:14:02 CEST 2025
Known bugs: 
1. Wurtzite.equivalent_directions(dir, drop_inverse=False)
    doesn't return exhaustive list. Example:
    ```python
    c = Wurtzite(3.25, 5.2)
    c.equivalent_directions('-1 -1 2 -2', drop_inverse=False)
    [[-1, 2, -1, -2], [-1, 2, -1, 2], [-1, -1, 2, -2], [-1, -1, 2, 2], [2, -1, -1, -2], [2, -1, -1, 2]]
    ```
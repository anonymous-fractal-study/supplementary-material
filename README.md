# Technologies Used

- Programming Language: C++
- Graphics / Visualization: olcPixelGameEngine

# How to Run

## Linux
```bash

# Example
git clone
make install
make gen
make run
```

## Windows
Using MSYS2 + MinGW-64
```bah
g++ -fopenmp Application.cpp \
    -luser32 -lgdi32 -lopengl32 -lgdiplus -lShlwapi -ldwmapi \
    -lstdc++fs -static -std=c++17 -O3 -mavx2 \
    -o app
```

Alternatively, you can simply run the **app** file.


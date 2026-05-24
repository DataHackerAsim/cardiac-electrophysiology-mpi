CC = mpicc
CFLAGS = -O3 -march=native -Wall -lm
TARGET = cardiac_sim
SRC = src/cardiac_sim.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) results/*.bin results/*.csv results/*.png

.PHONY: all clean

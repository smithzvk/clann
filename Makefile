CC =		gcc
OPT = 		-O3

.PHONY : clean
clean :
	rm libcmap.so

.PHONY : all
all : libcmap.so

libcmap.so : cmap.c
	$(CC) $(OPT) cmap.c -fPIC --shared -o libcmap.so -lm

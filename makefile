DEBUG =
CC = gcc
WARNINGS = -Wl,--no-as-needed -Wno-unused-result
ifdef DEBUG
        OPT = -g
else
        OPT = -O2
endif

FLAGS = $(OPT) $(WARNINGS) 
LIBS = -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

time_course: time_course.o
	$(CC) $(FLAGS) $+ -o $@ $(LIBS) 
time_course.o: time_course.c
	$(CC) -c $(FLAGS) $<
clean:
	rm -f *.o
	rm -f time_course

CC = g++
CXX = g++
CXXFLAGS = -Iinclude -I../libatlas_wrapper/include -I../anguso_arg_parser/include -I../libutility/include

MACHINE = $(shell uname -m)

SRC = hmmlite.cpp
OBJ = $(addprefix obj/$(MACHINE)/,$(SRC:.cpp=.o))

TARGET = lib/$(MACHINE)/libhmmlite.a

vpath %.cpp src
vpath %.o obj/$(MACHINE)
vpath %.a lib/$(MACHINE)

.PHONY: mk_machine_dir all clean allclean

all: CXXFLAGS:=-Wall -Werror -O2 $(CXXFLAGS)

all: mk_machine_dir $(TARGET)

debug: CXXFLAGS:=$(CXXFLAGS) -DDEBUG -g

debug: $(TARGET)

%.d: %.cpp
	$(CC) -M $(CXXFLAGS) $< > $@

lib/$(MACHINE)/libhmmlite.a: \
	obj/$(MACHINE)/hmmlite.o
	$(AR) rucs $@ $^

obj/$(MACHINE)/%.o: src/%.cpp
	$(CC) -c $(CXXFLAGS) -o $@ $^

mk_machine_dir:
	@mkdir -p obj/$(MACHINE)
	@mkdir -p lib/$(MACHINE)

allclean: clean
	$(RM) $(TARGET)

clean:
	$(RM) $(OBJ)

CC = g++

CFLAGS = -g -Wall -Werror -I/home/ugoc/include

LDFLAGS = -pthread \
					-lhmmlite \
					-latlas_wrapper \
					-latlas_wrapper_type \
					-llapack \
					-llapack_atlas \
					-lblas \
					-languso_arg_parser \
					-lfeature \
					-lutility

LDPATH = -L/home/ugoc/lib -L/usr/lib

main: main.cpp
	${CC} ${CFLAGS} $(LDPATH) $< ${LDFLAGS} -o $@

test_logarithmatics: test_logarithmatics.cpp
	${CC} ${CFLAGS} $(LDPATH) $< ${LDFLAGS} -o $@

clean:
	rm -f main

# Configuration pour MAC OSX
# Peut être modifié à votre convenance...
CXX = mpic++
CXXFLAGS = -std=c++17 -I/usr/local/include/
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -fsanitize=address -openmp
CXXFLAGS2 = CXXFLAGS
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall -openmp
CXXFLAGS += -O3 -march=native -Wall -openmp
endif
LIB=-L/usr/local/Cellar/sfml/2.5.1_2/lib -lsfml-graphics -lsfml-window -lsfml-system

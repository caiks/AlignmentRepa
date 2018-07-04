# AlignmentRepa

The AlignmentRepa repository is a fast Haskell implementation of some of the *practicable inducers* described in the paper *The Theory and Practice of Induction by Alignment* at http://greenlake.co.uk/. The AlignmentRepa repository depends on the [Alignment repository](https://github.com/caiks/Alignment) for the underlying *model* framework. The slower implementations of some of the *practicable inducers* in the Alignment repository can be used to verify the correctness of equivalent faster implementations in AlignmentRepa.

The AlignmentRepa repository uses high performance arrays. Single-dimensional arrays come from the [vector](http://hackage.haskell.org/package/vector) library. See [A Vector Tutorial](https://wiki.haskell.org/Numeric_Haskell:_A_Vector_Tutorial). Multi-dimensional shape polymorphic parallel arrays come from the [repa](http://hackage.haskell.org/package/repa) library. In addition, some compute-intensive array processing is implemented in C using the [Foreign Function Interface](https://wiki.haskell.org/Foreign_Function_Interface). See also [FFI](http://dev.stephendiehl.com/hask/#ffi) and [Data.Vector.Storable](http://hackage.haskell.org/package/vector-0.12.0.1/docs/Data-Vector-Storable.html).

The *induced models* are made persistent using the JSON format which is implemented in the [aeson](http://hackage.haskell.org/package/aeson) library.

There are a couple of useful libraries that should be installed along with repa and aeson to ensure consistent package versions -

[zlib](http://hackage.haskell.org/package/zlib): Compression and decompression in the gzip and zlib formats

[cassava](http://hackage.haskell.org/package/cassava): A CSV parsing and encoding library


## Installation

The AlignmentRepa module requires the [Haskell platform](https://www.haskell.org/downloads#platform) to be installed.

For example in Ubuntu,
```
sudo apt-get update
sudo apt-get install haskell-platform

apt-cache showpkg haskell-platform
2014.2.0.0.debian2

cabal update
cabal install repa repa-io vector-algorithms zlib cassava aeson aeson-pretty

```
Then download the zip file or use git to get the repository and the underlying Alignment repository -
```
cd
git clone https://github.com/caiks/Alignment.git
git clone https://github.com/caiks/AlignmentRepa.git
```

## Usage

Typically we wish to force compilation in ghci. See [GHCi Performance](http://dev.stephendiehl.com/hask/#ghci).
Load AlignmentDevRepa to import the modules and define various useful abbreviated functions,
```
cd AlignmentRepa
gcc -fPIC -c AlignmentForeign.c -o AlignmentForeign.o -O3
ghci -i../Alignment -i../AlignmentRepa ../AlignmentRepa/AlignmentForeign.o
```
```hs
:set -fobject-code
:set +m
:l AlignmentDevRepa

let aa = regdiag 2 2

rp $ aa

aaar (sys aa) aa
```

## Documentation

Note that a discussion of the implementation of higher performance *inducers* is to follow.


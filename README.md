# AlignmentRepa

The AlignmentRepa repository is a fast Haskell and C implementation of some of the *practicable inducers* described in the paper *The Theory and Practice of Induction by Alignment* at https://greenlake.co.uk/. The AlignmentRepa repository depends on the [Alignment repository](https://github.com/caiks/Alignment) for the underlying *model* framework. The slower implementations of some of the *practicable inducers* in the Alignment repository can be used to verify the correctness of equivalent faster implementations in AlignmentRepa.

The AlignmentRepa repository uses high performance arrays. Single-dimensional arrays are defined in the [vector](http://hackage.haskell.org/package/vector) library. See [Numeric Haskell](https://wiki.haskell.org/Numeric_Haskell:_A_Vector_Tutorial). Multi-dimensional shape polymorphic parallel arrays are defined in the [repa](http://hackage.haskell.org/package/repa) library. In addition, some compute-intensive array processing is implemented in C using the [Foreign Function Interface](https://wiki.haskell.org/Foreign_Function_Interface). See also [FFI](http://dev.stephendiehl.com/hask/#ffi) and [Data.Vector.Storable](http://hackage.haskell.org/package/vector-0.12.0.1/docs/Data-Vector-Storable.html).

The *induced models* are made persistent using the JSON format which is implemented in the [aeson](http://hackage.haskell.org/package/aeson) library.

There are a couple of useful libraries that should be installed along with repa and aeson to ensure consistent package versions:

[zlib](http://hackage.haskell.org/package/zlib): Compression and decompression in the gzip and zlib formats

[cassava](http://hackage.haskell.org/package/cassava): A CSV parsing and encoding library

## Documentation

The [Haskell implementation of fast Practicable Inducers](https://greenlake.co.uk/pages/inducer_haskell_impl_repa) discusses the implementation of the *inducers* using this repository. 

## Install

The `AlignmentRepa` module requires the [Haskell platform](https://www.haskell.org/downloads#platform) to be installed. The project is managed using [stack](https://docs.haskellstack.org/en/stable/).

Download the zip files or use git to get the AlignmentRepa repository and the underlying Alignment repository -
```
cd
git clone https://github.com/caiks/Alignment.git
git clone https://github.com/caiks/AlignmentRepa.git
```

Then build with the following -
```
cd ~/AlignmentRepa
stack build --ghc-options -w

```

## Usage

Use `stack ghci` or `stack repl` for a run-eval-print loop (REPL) environment. 
Load `AlignmentDevRepa` to import the modules and define various useful abbreviated functions,
```sh
cd ~/AlignmentRepa
stack ghci --ghci-options -w

```
```hs
:set -fobject-code
:set +m
:l AlignmentDevRepa

let aa = regdiag 2 2

rp $ aa
"{({(1,1),(2,1)},1 % 1),({(1,2),(2,2)},1 % 1)}"

aa
Histogram (fromList [(State (fromList [(VarInt 1,ValInt 1),(VarInt 2,ValInt 1)]),1 % 1),(State (fromList [(VarInt 1,ValInt 2),(VarInt 2,ValInt 2)]),1 % 1)])

aaar (sys aa) aa
HistogramRepa {histogramRepasVectorVar = [VarInt 1,VarInt 2], histogramRepasMapVarInt = fromList [(VarInt 1,0),(VarInt 2,1)], histogramRepasArray = AUnboxed [2,2] [1.0,0.0,0.0,1.0]}
```
Note that some modules may become [unresolved](https://downloads.haskell.org/~ghc/7.10.3-rc1/users_guide/ghci-obj.html), for example,
```hs
rp $ Set.fromList [1,2,3]

<interactive>:9:1: Not in scope: ‘Set.fromList’
```
or 
```hs
rp $ fudEmpty

<interactive>:10:6: Not in scope: ‘fudEmpty’
```
In this case, re-import the modules explicitly as defined in `AlignmentDevRepa`, for example,
```hs
import qualified Data.Set as Set
import qualified Data.Map as Map
import Alignment

rp $ Set.fromList [1,2,3]
"{1,2,3}"

rp $ fudEmpty
"{}"
```




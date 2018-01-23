{-# LANGUAGE OverloadedStrings #-}

module AlignmentAesonPretty (
  historyPersistentsEncode,
  historyPersistentsEncodePrefixed,
  transformPersistentsEncodePrefixed,
  fudPersistentsEncode,
  fudPersistentsEncodePrefixed,
  decompFudsPersistentsEncode
)
where
import Data.List as List
import qualified Data.Set as Set
import qualified Data.Map as Map
import qualified Data.Vector as Vector
import Control.Applicative
import Control.Monad
import Data.Aeson hiding (Value)
import Data.ByteString.Lazy.Char8
import Data.ByteString.Builder
import Data.Monoid
import GHC.Real
import GHC.Generics
import AlignmentUtil
import Alignment hiding (derived)
import AlignmentAeson

historyPersistentsEncode :: HistoryPersistent -> ByteString
historyPersistentsEncode hh =
  mconcat $ ["{\n\t\"hsystem\":[\n"] ++ 
    List.intersperse ",\n" ["\t\t" <> Data.Aeson.encode vv | let SystemPersistent uu = hsystem hh, vv <- uu] ++ 
    ["\n\t],\n\t\"hstates\":[\n"] ++ 
    List.intersperse ",\n" ["\t\t" <> Data.Aeson.encode ss | ss <- hstates hh] ++ ["\n\t]\n}"]

historyPersistentsEncodePrefixed :: Int -> HistoryPersistent -> ByteString
historyPersistentsEncodePrefixed i hh =
  mconcat $ ["{\n",p,"\t\"hsystem\":[\n"] ++ 
    List.intersperse ",\n" [p <> "\t\t" <> Data.Aeson.encode vv | let SystemPersistent uu = hsystem hh, vv <- uu] ++ 
    ["\n",p,"\t],\n",p,"\t\"hstates\":[\n"] ++ 
    List.intersperse ",\n" [p <> "\t\t" <> Data.Aeson.encode ss | ss <- hstates hh] ++ ["\n",p,"\t]\n",p,"}"]
  where
    p = pack $ List.replicate i '\t'

transformPersistentsEncodePrefixed :: Int -> TransformPersistent -> ByteString
transformPersistentsEncodePrefixed i hh =
  mconcat $ ["{\n", p, "\t\"derived\":["] ++ List.intersperse "," (List.map Data.Aeson.encode (derived hh)) ++ ["],\n", p, "\t\"history\":", historyPersistentsEncodePrefixed (i+1) (history hh), "\n", p, "}"]
  where
    p = pack $ List.replicate i '\t'

fudPersistentsEncode :: FudPersistent -> ByteString
fudPersistentsEncode = fudPersistentsEncodePrefixed 0

fudPersistentsEncodePrefixed :: Int -> FudPersistent -> ByteString
fudPersistentsEncodePrefixed i (FudPersistent ff) =
  mconcat $ ["[\n"] ++ 
    List.intersperse ",\n" [p <> "\t" <> transformPersistentsEncodePrefixed (i+1) tt | tt <- ff] ++ 
    ["\n", p, "]"]
  where
    p = pack $ List.replicate i '\t'

decompFudsPersistentsEncode :: DecompFudPersistent -> ByteString
decompFudsPersistentsEncode df =
  mconcat $ ["{\n\t\"paths\":[\n"] ++ 
    List.intersperse ",\n" ["\t\t" <> Data.Aeson.encode ss | ss <- paths df] ++ 
    ["\n\t],\n\t\"nodes\":[\n"] ++ 
    List.intersperse ",\n" ["\t[" <> historyPersistentsEncodePrefixed 2 hh <> ",\n\t" <> fudPersistentsEncodePrefixed 1 ff <> "]" | (hh,ff) <- nodes df] ++ 
    ["\t]\n}"]


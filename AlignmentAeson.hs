{-# LANGUAGE OverloadedStrings, DeriveGeneric, RankNTypes #-}

module AlignmentAeson (
  VariablePersistent(..),
  SystemPersistent(..),
  HistoryPersistent(..),
  HistogramPersistent(..),
  TransformPersistent(..),
  FudPersistent(..),
  DecompFudPersistent(..),
  stringsVariable,
  stringsValue,
  persistentsSystem,
  systemsPersistent,
  persistentsHistory,
  historiesPersistent,
  persistentHistorysSystem,
  persistentsHistogram,
  histogramsPersistent,
  persistentsTransform,
  transformsPersistent,
  persistentsFud,
  fudsPersistent,
  persistentFudsSystem,
  persistentsDecompFud,
  decompFudsPersistent,
  persistentDecompFudsSystem
)
where
import Data.List as List
import qualified Data.Set as Set
import qualified Data.Map as Map
import Control.Applicative
import Control.Monad
import Data.Aeson hiding (Value)
import Data.Monoid
import GHC.Real
import GHC.Generics
import AlignmentUtil
import Alignment hiding (derived)

data VariablePersistent = VariablePersistent { var :: String, values :: [String] } deriving (Show,Generic) 

instance FromJSON VariablePersistent
instance ToJSON VariablePersistent

data SystemPersistent = SystemPersistent [VariablePersistent] deriving (Show,Generic) 

instance FromJSON SystemPersistent
instance ToJSON SystemPersistent

data HistoryPersistent = HistoryPersistent { hsystem :: SystemPersistent, hstates :: [[Int]] } deriving (Show,Generic) 

instance FromJSON HistoryPersistent
instance ToJSON HistoryPersistent

data HistogramPersistent = HistogramPersistent { asystem :: SystemPersistent, astates :: [([Int],String)] } deriving (Show,Generic) 

instance FromJSON HistogramPersistent
instance ToJSON HistogramPersistent

data TransformPersistent = TransformPersistent { history :: HistoryPersistent, derived :: [String] } deriving (Show,Generic) 

instance FromJSON TransformPersistent
instance ToJSON TransformPersistent

data FudPersistent = FudPersistent [TransformPersistent] deriving (Show,Generic) 

instance FromJSON FudPersistent
instance ToJSON FudPersistent

data DecompFudPersistent = DecompFudPersistent { nodes :: [(HistoryPersistent,FudPersistent)], paths :: [[Int]] } 
                                                                                                    deriving (Show,Generic) 

instance FromJSON DecompFudPersistent
instance ToJSON DecompFudPersistent

stringsVariable :: String -> Variable
stringsVariable s = 
  if ll /= [] && snd (head ll) == "" 
    then VarInt (fst (head ll)) 
    else 
      if length s >= 2 && head s == '<' && last s == '>' && any (== ',') s
        then VarPair (stringsVariable (start s), stringsVariable (end s))
        else VarStr s
  where
    ll = reads s :: [(Integer, String)]
    start = tail . init . reverse . dropWhile (/= ',') . reverse . init 
    end = reverse . takeWhile (/= ',') . reverse . init 

stringsValue :: String -> Value
stringsValue s = if isDouble then ValDouble d else (if isInt then ValInt i else ValStr s)
  where
    ll = reads s :: [(Integer, String)]
    i = fst (head ll)
    isInt = ll /= [] && snd (head ll) == ""
    lld = reads s :: [(Double, String)]
    d = if s == "Infinity" then (1/0) else if s == "-Infinity" then (-1/0) else fst (head lld)
    isDouble  = s == "Infinity" || s == "-Infinity" || (lld /= [] && snd (head lld) == "" && (find (=='.') s /= Nothing))

stringsRational :: String -> Rational
stringsRational s 
  | ll /= [] && snd (head ll) == "" = fst (head ll)
  | mm /= [] && snd (head mm) == "" = toRational (fst (head mm))
  | otherwise = 0
  where
    ll = reads s :: [(Rational, String)]
    mm = reads s :: [(Double, String)]

rationalsString :: Rational -> String
rationalsString r 
  | denominator r == 1 = show $ numerator r
  | otherwise = show (numerator r) ++ "%" ++ show (denominator r)

persistentsSystem :: SystemPersistent -> Maybe System
persistentsSystem (SystemPersistent uu') = lluu $ [(svar (var v'), llqq [sval w' | w' <- values v']) | v' <- uu']
  where
    lluu = listsSystem
    svar = stringsVariable
    sval = stringsValue        
    llqq = Set.fromList

systemsPersistent :: System -> SystemPersistent
systemsPersistent uu = SystemPersistent [VariablePersistent {var = vars v, values = map vals (qqll ww)} | (v,ww) <- uull uu]
  where
    uull = systemsList
    vars (VarPair (v,w)) = "<" ++ vars v ++ "," ++ vars w ++ ">"
    vars (VarInt i) = show i
    vars (VarStr s) = s
    vals (ValDouble d) = show d
    vals (ValInt i) = show i
    vals (ValStr s) = s
    qqll = Set.toList

persistentsHistory :: HistoryPersistent -> Maybe History
persistentsHistory hh 
  | uu' == Nothing = Nothing
  | otherwise = llhh $ [(IdInt i, llss [(v,w) | (j,k) <- zip [0 .. length ll -1] ss, let v = ll !! j, 
                        let ww = mm Map.! v, k < length ww, let w = ww !! k]) | (i,ss) <- zip [1..] (hstates hh)]
  where
    uu' = ppuu $ hsystem hh
    SystemPersistent xx = hsystem hh
    ll = [svar (var x) | x <- xx]
    mm = llmm [(svar (var x), map sval (values x)) | x <-xx]
    llhh = listsHistory
    llss = listsState
    ppuu = persistentsSystem
    svar = stringsVariable
    sval = stringsValue        
    llmm = Map.fromList

persistentHistorysSystem :: HistoryPersistent -> Maybe System
persistentHistorysSystem = persistentsSystem . hsystem

historiesPersistent :: History -> HistoryPersistent
historiesPersistent hh = 
    HistoryPersistent {hsystem = uupp (hsys hh), hstates = [[mm Map.! v Map.! (ss `sat` v) | v <- hvarsll hh] | ss <- hhll hh]}
  where
    mm = llmm [(v, llmm (zip (qqll ww) [0..])) | (v,ww) <- uull (hsys hh)]
    uupp = systemsPersistent
    hsys = histogramsSystemImplied . historiesHistogram
    hvarsll = qqll . historiesSetVar
    hhll = snd . unzip . historiesList
    uull = systemsList
    sat ss v = fromJust $ statesVarsValue ss v
    qqll = Set.toList
    llmm :: forall k a. Ord k => [(k, a)] -> Map.Map k a
    llmm = Map.fromList

persistentsHistogram :: HistogramPersistent -> Maybe Histogram
persistentsHistogram aa
  | uu' == Nothing = Nothing
  | otherwise = llaa $ [(llss [(v,w) | (j,k) <- zip [0 .. length ll -1] ss, let v = ll !! j, 
                  let ww = mm Map.! v, k < length ww, let w = ww !! k], sr r) | (ss,r) <- astates aa]
  where
    uu' = ppuu $ asystem aa
    SystemPersistent xx = asystem aa
    ll = [svar (var x) | x <- xx]
    mm = llmm [(svar (var x), map sval (values x)) | x <-xx]
    llaa = listsHistogram
    llss = listsState
    ppuu = persistentsSystem
    svar = stringsVariable
    sval = stringsValue        
    sr = stringsRational
    llmm = Map.fromList

histogramsPersistent :: Histogram -> HistogramPersistent
histogramsPersistent aa = 
    HistogramPersistent {asystem = uupp (sys aa), 
      astates = [([mm Map.! v Map.! (ss `sat` v) | v <- varsll aa], rs c) | (ss,c) <- aall aa]}
  where
    mm = llmm [(v, llmm (zip (qqll ww) [0..])) | (v,ww) <- uull (sys aa)]
    uupp = systemsPersistent
    sys = histogramsSystemImplied
    varsll = qqll . histogramsVars
    aall = histogramsList
    uull = systemsList
    sat ss v = fromJust $ statesVarsValue ss v
    rs = rationalsString
    qqll = Set.toList
    llmm :: forall k a. Ord k => [(k, a)] -> Map.Map k a
    llmm = Map.fromList

persistentsTransform :: TransformPersistent -> Maybe Transform
persistentsTransform tt
  | hh' == Nothing = Nothing
  | otherwise = trans (unit (hhaa (fromJust hh'))) (llqq (map svar (derived tt)))
  where
    hh' = pphh $ history tt
    pphh = persistentsHistory
    trans = histogramsSetVarsTransform
    unit = histogramsUnit
    hhaa = historiesHistogram
    svar = stringsVariable
    llqq = Set.fromList

transformsPersistent :: Transform -> TransformPersistent
transformsPersistent tt = 
    TransformPersistent {history = hhpp (aahh (his tt)), derived = map vs (qqll (der tt))} 
  where
    his = transformsHistogram
    der = transformsDerived
    hhpp = historiesPersistent
    aahh aa = fromJust $ histogramsHistory $ unit aa
    unit = histogramsUnit
    vs (VarPair (v,w)) = "<" ++ vs v ++ "," ++ vs w ++ ">"
    vs (VarInt i) = show i
    vs (VarStr s) = s
    qqll = Set.toList

persistentsFud :: FudPersistent -> Maybe Fud
persistentsFud (FudPersistent ll)
  | mm' == Nothing = Nothing
  | otherwise = qqff (llqq (fromJust mm'))
  where
    mm' = mapM pptt ll
    pptt = persistentsTransform
    qqff = setTransformsFud
    llqq = Set.fromList

persistentFudsSystem :: FudPersistent -> Maybe System
persistentFudsSystem (FudPersistent ll)
  | mm' == Nothing = Nothing
  | otherwise = Just $ lluu $ fromJust mm'
  where
    mm' = mapM (ppuu . hsystem . history) ll
    ppuu = persistentsSystem
    lluu = foldl pairSystemsUnion systemEmpty

fudsPersistent :: Fud -> FudPersistent
fudsPersistent ff = FudPersistent (map ttpp (qqll (ffqq ff)))
  where
    ffqq = fudsSetTransform
    ttpp = transformsPersistent
    qqll = Set.toList

persistentsDecompFud :: DecompFudPersistent -> Maybe DecompFud
persistentsDecompFud df
  | hh' == Nothing = Nothing
  | ff' == Nothing = Nothing
  | otherwise = zzdf zz
  where
    hh' = mapM pphh $ fst $ unzip $ nodes df
    ff' = mapM ppff $ snd $ unzip $ nodes df
    nn = zip (map (qqmin . hhqq) (fromJust hh')) (fromJust ff')
    zz = funcsTreesMap (\i -> nn !! i) $ pathsTree $ llqq $ paths df
    pphh = persistentsHistory
    ppff = persistentsFud
    zzdf = treePairStateFudsDecompFud 
    hhqq = historiesSetState
    qqmin qq = if qq /= Set.empty then Set.findMin qq else stateEmpty
    qqff = setTransformsFud
    llqq = Set.fromList

decompFudsPersistent :: DecompFud -> DecompFudPersistent
decompFudsPersistent df = DecompFudPersistent {nodes = nn, paths = pp} 
  where
    zz = fst $ funcsListsTreesTraversePreOrder (flip (,)) [0..] $ dfzz df
    nn = [(hhpp (sshh ss), ffpp ff) | (ss,ff) <- snd $ unzip $ sort $ qqll $ treesElements zz]
    pp = Set.toList $ treesPaths $ funcsTreesMap fst zz
    ffpp = fudsPersistent
    hhpp = historiesPersistent
    dfzz = decompFudsTreePairStateFud
    sshh ss = fromJust $ histogramsHistory $ fromJust $ setStatesHistogramUnit $ Set.singleton ss
    qqll = Set.toList

persistentDecompFudsSystem :: DecompFudPersistent -> Maybe System
persistentDecompFudsSystem df
  | mm' == Nothing = Nothing
  | otherwise = Just $ lluu $ fromJust mm'
  where
    mm' = mapM ppuu $ snd $ unzip $ nodes df
    ppuu = persistentFudsSystem
    lluu = foldl pairSystemsUnion systemEmpty



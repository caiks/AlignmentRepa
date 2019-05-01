{-# LANGUAGE RankNTypes, BangPatterns #-}

module AlignmentRepa (
  HistogramRepa(..),
  HistogramRepaRed(..),
  HistogramRepaVec(..),
  HistogramRepaRedVec(..),
  HistoryRepa(..),
  TransformRepa(..),
  histogramRepaEmpty,
  histogramRepaVecEmpty,
  arraysHistogramRepa,
  histogramRepasSystem,
  vectorHistogramRepasHistogramRepaVec,
  vectorHistogramRepasHistogramRepaVec_u,
  histogramRepaVecsVectorHistogramRepas,
  vectorHistogramRepaRedsHistogramRepaRedVec_u,
  histogramRepaRedVecsVectorHistogramRepaReds,
  systemsHistogramsHistogramRepa,
  systemsHistogramRepasHistogram,
  histogramRepaVecsSum,
  histogramRepaVecsFaclnsRepaVecs,
  setSetVarsHistogramRepasPartition_u,
  setSetVarsHistogramRepaVecsPartitionVec_u,
  setSetVarsHistogramRepaVecsPartitionVec_u_1,
  varsHistogramRepaVecsRollVec_u,
  sumsHistogramRepa4VecsRollMapPairsHistogramRepa4VecsSum_u,
  sumsHistogramRepa4VecsRollMapPairsHistogramRepa4VecsSum_u_1,
  sumsHistogramRepasRollMapPairsHistogramRepasSum_u,
  varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u,
  varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u_1,
  histogramRepaVecsRollMax,
  setVarsHistogramRepasReduce,
  setVarsHistogramRepasReduce_1,
  setVarsHistogramRepasReduce_2,
  setVarsHistogramRepasReduce_3,
  setVarsHistogramRepasReduce_4,
  varsHistogramRepaVecsReduceSingleVec_u,
  varsHistogramRepa4VecsReduceSingle_u,
  histogramRepasRed_u,
  histogramRepaVecsRedVec,
  histogramRepa4VecsRed_u,
  varsHistogramRepaRedVecsSingleVec_u,
  varsHistogramRepaRedsSingle_u,
  setVarsHistogramRepaRedsRed,
  setSetVarsHistogramRepasPartitionRed_u,
  setSetVarsHistogramRepasPartitionRed_u_1,
  setSetVarsHistogramRepaVecsPartitionRedVec_u,
  setSetVarsHistogramRepaVecsPartitionRedVec_u_1,
  histogramRepaRedsIndependent,
  histogramRepaRedsIndependent_1,
  histogramRepaRedVecsIndependent_u,
  histogramRepaRedVecsIndependent_u_1,
  setSetVarsHistogramRepaVecsPartitionIndependentVec_u,
  setSetVarsHistogramRepaPairStorablesPartitionIndependentPair_u,
  parametersHistogramRepaVecsSetTuplePartitionTop_u,
  parametersHistogramRepaVecsSetTuplePartitionTopByM_u,
  historyRepaEmpty,
  arraysHistoryRepa_u,
  arraysHistoryRepaCardinal_u,
  vectorHistoryRepasConcat_u,
  systemsHistoriesHistoryRepa,
  systemsHistoriesHistoryRepa_u,
  systemsHistoryRepasHistory_u,
  historyRepasSize,
  historyRepasDimension,
  historyRepasSetVariable,
  setVarsHistoryRepasCountApproxs,
  setVarsHistoryRepasReduce,
  setVarsHistoryRepasReduce_1,
  setVarsHistoryRepasReduce_2,
  setVarsHistoryRepasReduce_3,
  setVarsHistoryRepaStorablesReduce,
  historyRepasTransformRepasApply_u,
  historyRepasListTransformRepasApply,
  listVariablesListTransformRepasSort,
  historyRepasListTransformRepasApply_u,
  systemsFudsHistoryRepasMultiply,
  systemsFudsHistoryRepasMultiply_u,
  systemsDecompFudsHistoryRepasMultiply,
  systemsDecompFudsHistoryRepasMultiply_r,
  systemsDecompFudsHistoryRepasSetVariablesListHistogramLeaf,
  systemsDecompFudsHistoryRepasHistoriesQuery,
  systemsDecompFudsHistoryRepasHistoryRepasQuery,
  systemsDecompFudMultipliesHistoryRepasHistoriesQuery,
  systemsDecompFudMultipliesHistoryRepasHistoryRepasQuery,
  systemsDecompFudMultipliesHistoryRepasHistoryRepasQueryAny,
  systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest,
  systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_1,
  systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_2,
  systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_3,
  systemsDecompFudsHistoryRepasDecompFudReduced,
  historyRepasRed,
  setVarsHistoryRepasRed,
  setVarsHistoryRepasHistoryRepaReduced,
  eventsHistoryRepasHistoryRepaSelection,
  historyRepasHistoryRepasHistoryRepaSelection_u,
  historyRepasListsList,
  systemsListVariablesListsListsHistoryRepa,
  systemsListVariablesListsListsHistoryRepa_u,
  systemsTransformsTransformRepa,
  systemsTransformsTransformRepa_u,
  vectorPairsTop,
  parametersSetVarsHistoryRepasSetSetVarsAlignedTop,
  parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u,
  parametersSetVarsHistoryRepasSetSetVarsAlignedTop_1,
  parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_1,
  parametersSetVarsHistoryRepasSetSetVarsAlignedTop_2,
  parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_2,
  parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop,
  parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u,
  parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_1,
  parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u_1,
  parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u_2,
  parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop,
  parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop_u,
  parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedExcludeHiddenDenseTop_u
)
where
import Control.Monad
import Control.Monad.ST
import Data.STRef
import Data.Int
import Data.List as List
import qualified Data.Set as Set
import qualified Data.Map as Map
import qualified Data.IntMap as IntMap
import qualified Data.Vector as V
import qualified Data.Vector.Algorithms.Merge as VA
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SMV
import Data.Array.Repa as R
import GHC.Real
import Foreign.Ptr
import Foreign.C.Types
import System.IO.Unsafe
import AlignmentRepaVShape
import AlignmentUtil
import Alignment
import AlignmentApprox

data HistogramRepa = HistogramRepa {
  histogramRepasVectorVar :: !(V.Vector Variable),
  histogramRepasMapVarInt :: Map.Map Variable Int,
  histogramRepasArray :: !(Array U VShape Double)}
               deriving (Eq, Read, Show)

instance Ord HistogramRepa where
  compare _ _ = EQ

data HistogramRepaVec = HistogramRepaVec {
  histogramRepaVecsVectorVar :: !(V.Vector Variable),
  histogramRepaVecsMapVarInt :: Map.Map Variable Int,
  histogramRepaVecsSize :: !Double,
  histogramRepaVecsShape :: !VShape,
  histogramRepaVecsArray :: !(V.Vector (UV.Vector Double))}
               deriving (Eq, Read, Show)

instance Ord HistogramRepaVec where
  compare _ _ = EQ

data HistogramRepaRed = HistogramRepaRed {
  histogramRepaRedsVectorVar :: !(V.Vector Variable),
  histogramRepaRedsMapVarInt :: Map.Map Variable Int,
  histogramRepaRedsShape :: !VShape,
  histogramRepaRedsVectorArray :: !(V.Vector (UV.Vector Double))}
               deriving (Eq, Read, Show)

instance Ord HistogramRepaRed where
  compare _ _ = EQ

data HistogramRepaRedVec = HistogramRepaRedVec {
  histogramRepaRedVecsVectorVar :: !(V.Vector Variable),
  histogramRepaRedVecsMapVarInt :: Map.Map Variable Int,
  histogramRepaRedVecsSize :: !Double,
  histogramRepaRedVecsShape :: !VShape,
  histogramRepaRedVecsVectorArray :: !(V.Vector (V.Vector (UV.Vector Double)))}
               deriving (Eq, Read, Show)

data HistoryRepa = HistoryRepa {
  historyRepasVectorVar :: !(V.Vector Variable),
  historyRepasMapVarInt :: Map.Map Variable Int,
  historyRepasShape :: !VShape,
  historyRepasArray :: !(Array U DIM2 Int16)}
               deriving (Eq, Read, Show)

instance Ord HistoryRepa where
  compare _ _ = EQ

data TransformRepa = TransformRepa {
  transformRepasVectorVar :: !(V.Vector Variable),
  transformRepasMapVarInt :: Map.Map Variable Int,
  transformRepasVarDerived :: !Variable,
  transformRepasValency :: !Int16,
  transformRepasArray :: !(Array U VShape Int16)}
               deriving (Eq, Read, Show)

instance Ord TransformRepa where
  compare _ _ = EQ

histogramRepaEmpty :: HistogramRepa
histogramRepaEmpty = HistogramRepa vempty mempty (llrr vsempty [])
  where
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    mempty = Map.empty
    vempty = V.empty

histogramRepaVecEmpty :: HistogramRepaVec
histogramRepaVecEmpty = HistogramRepaVec vempty mempty 0 vsempty vempty
  where
    vsempty = UV.empty
    mempty = Map.empty
    vempty = V.empty

arraysHistogramRepa :: Array U VShape Double -> HistogramRepa
arraysHistogramRepa rr = HistogramRepa (llvv n vv) mvv rr
  where 
    n = rank (extent rr)
    vv = List.map VarIndex [0 .. n-1]
    mvv = llmm (zip vv [0..])
    llmm = Map.fromList
    llvv = V.fromListN

vectorHistogramRepasHistogramRepaVec :: Double -> V.Vector HistogramRepa -> Maybe HistogramRepaVec
vectorHistogramRepasHistogramRepaVec z vrr 
  | V.null vrr = Nothing
  | not $ V.and $ V.map (\(HistogramRepa _ _ ss) -> extent ss == svv) vrr = Nothing
  | otherwise = Just $ vrrrrv z vrr
  where
    HistogramRepa vvv mvv rr = vrr V.! 0
    svv = extent rr
    vrrrrv = vectorHistogramRepasHistogramRepaVec_u

vectorHistogramRepasHistogramRepaVec_u :: Double -> V.Vector HistogramRepa -> HistogramRepaVec
vectorHistogramRepasHistogramRepaVec_u z vrr = HistogramRepaVec vvv mvv z svv (V.map rraa vrr)
  where
    HistogramRepa vvv mvv rr = vrr V.! 0
    svv = extent rr
    rraa = R.toUnboxed . histogramRepasArray

histogramRepaVecsVectorHistogramRepas :: HistogramRepaVec -> V.Vector HistogramRepa
histogramRepaVecsVectorHistogramRepas rrv = V.map (\aa -> HistogramRepa vvv mvv (R.fromUnboxed svv aa)) vaa
  where
    HistogramRepaVec vvv mvv _ svv vaa = rrv

vectorHistogramRepaRedsHistogramRepaRedVec_u :: Double -> V.Vector HistogramRepaRed -> HistogramRepaRedVec
vectorHistogramRepaRedsHistogramRepaRedVec_u z vrr = HistogramRepaRedVec vvv mvv z svv (V.map rraa vrr)
  where
    HistogramRepaRed vvv mvv svv rr = vrr V.! 0
    rraa = histogramRepaRedsVectorArray

histogramRepaRedVecsVectorHistogramRepaReds :: HistogramRepaRedVec -> V.Vector HistogramRepaRed
histogramRepaRedVecsVectorHistogramRepaReds rrv = V.map (\aa -> HistogramRepaRed vvv mvv svv aa) vaa
  where
    HistogramRepaRedVec vvv mvv z svv vaa = rrv

histogramRepasSystem :: HistogramRepa -> System
histogramRepasSystem aa = lluu [(v, llqq (List.map ValIndex [0 .. sh UV.! i - 1])) | (i,v) <- zip [0..] (vvll vv)]
  where 
    vv = histogramRepasVectorVar aa
    rr = histogramRepasArray aa
    sh = extent rr
    lluu = listsSystem_u
    vvll = V.toList
    llqq = Set.fromList

systemsHistogramsHistogramRepa :: System -> Histogram -> Maybe HistogramRepa
systemsHistogramsHistogramRepa uu aa
  | aa /= empty && (vars aa `subset` uvars uu) = 
      Just $ HistogramRepa (llvv vv) mvv (llrr sh (elems nn))
  | otherwise = Nothing
  where 
    vv = qqll $ vars aa
    mvv = llmm (zip vv [0..])
    mm = llmm [(v, llmm (zip (qqll ww) [0..])) | v <- vv, let ww = uvals uu v]
    sh = shapeOfList [Map.size (mm Map.! v) | v <- vv] :: VShape
    nn = llim (+) $ [(i,0) | i <- [0 .. size sh - 1]] List.++ 
           [(toIndex sh (shapeOfList [mm Map.! v Map.! (ss `sat` v) | v <- vv] :: VShape), fromRational c) | 
             (ss,c) <- aall aa]
    empty = histogramEmpty
    aall = histogramsList
    vars = histogramsVars
    sat ss v = fromJust $ statesVarsValue ss v
    uvals uu v = fromJust $ systemsVarsSetValue uu v
    uvars = systemsVars
    uull = systemsList
    llrr = R.fromListUnboxed
    elems = IntMap.elems
    llim = IntMap.fromListWith
    llmm :: forall k a. Ord k => [(k, a)] -> Map.Map k a
    llmm = Map.fromList
    qqll = Set.toList
    subset = Set.isSubsetOf
    llvv = V.fromList

systemsHistogramRepasHistogram :: System -> HistogramRepa -> Maybe Histogram
systemsHistogramRepasHistogram uu aa
  | vvqq vv `subset` uvars uu && sh == sh' = 
      llaa [(llss [(v,w) | (j,k) <- zip [0..] ss, let v = vv V.! j, let w = mm Map.! v V.! k], toRational c) | 
             (i,c) <- zip [0..] (rrll rr), let ss = listOfShape (fromIndex sh i)]
  | otherwise = Nothing
  where 
    vv = histogramRepasVectorVar aa
    rr = histogramRepasArray aa
    mm = llmm [(v, llvv (qqll ww)) | v <- V.toList vv, let ww = uvals uu v]
    sh = shapeOfList [V.length (mm Map.! v) | v <- vvll vv] :: VShape
    sh' = extent rr
    llaa = listsHistogram
    llss = listsState
    uvals uu v = fromJust $ systemsVarsSetValue uu v
    uvars = systemsVars
    uull = systemsList
    rrll = R.toList
    llmm = Map.fromList
    qqll = Set.toList
    vvqq = Set.fromList . V.toList
    subset = Set.isSubsetOf
    llvv = V.fromList
    vvll = V.toList

historyRepaEmpty :: HistoryRepa
historyRepaEmpty = HistoryRepa vempty mempty vsempty (llrr (Z :. 0 :. 0) [])
  where
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    mempty = Map.empty
    vempty = V.empty

arraysHistoryRepa_u :: VShape -> Array U DIM2 Int16 -> HistoryRepa
arraysHistoryRepa_u svv rr = HistoryRepa (llvv n vv) mvv svv rr
  where 
    Z :. n :. z = extent rr
    vv = List.map VarIndex [0 .. n-1]
    mvv = llmm (zip vv [0..])
    llmm = Map.fromList
    llvv = V.fromListN

arraysHistoryRepaCardinal_u :: VShape -> Array U DIM2 Int16 -> HistoryRepa
arraysHistoryRepaCardinal_u svv rr = HistoryRepa (llvv n vv) mvv svv rr
  where 
    Z :. n :. z = extent rr
    vv = List.map (VarInt .toInteger) [1 .. n]
    mvv = llmm (zip vv [0..])
    llmm = Map.fromList
    llvv = V.fromListN

systemsHistoriesHistoryRepa :: System -> History -> Maybe HistoryRepa
systemsHistoriesHistoryRepa uu hh
  | hh /= empty && (vars hh `subset` uvars uu) = 
      Just $ HistoryRepa (llvv vv) mvv (llvu sh) (computeS (R.transpose (llrr sh' nn)))
  | otherwise = Nothing
  where 
    vv = qqll $ vars hh
    mvv = llmm (zip vv [0..])
    sh = [uval uu v | v <- vv]
    mm = llmm [(v, llmm (zip (qqll ww) [0..])) | v <- vv, let ww = uvals uu v]
    sh' = Z :. size hh :. card (vars hh)
    nn = [mm Map.! v Map.! u | (_,ss) <- hhll hh, (v,u) <- ssll ss]
    size = fromInteger . historiesSize
    empty = historyEmpty
    hhll = historiesList
    vars = historiesVars
    ssll = statesList
    uvals uu v = fromJust $ systemsVarsSetValue uu v
    uval uu v = card $ fromJust $ systemsVarsSetValue uu v
    uvars = systemsVars
    uull = systemsList
    llrr = R.fromListUnboxed
    llmm :: forall k a. Ord k => [(k, a)] -> Map.Map k a
    llmm = Map.fromList
    card = Set.size
    qqll = Set.toList
    subset = Set.isSubsetOf
    llvv = V.fromList
    llvu = UV.fromList

systemsHistoriesHistoryRepa_u :: System -> History -> HistoryRepa
systemsHistoriesHistoryRepa_u uu hh = HistoryRepa (llvv vv) mvv (llvu sh) (computeS (R.transpose (llrr sh' nn)))
  where 
    vv = qqll $ vars hh
    mvv = llmm (zip vv [0..])
    sh = [uval uu v | v <- vv]
    mm = llmm [(v, llmm (zip (qqll ww) [0..])) | v <- vv, let ww = uvals uu v]
    sh' = Z :. size hh :. card (vars hh)
    nn = [mm Map.! v Map.! u | (_,ss) <- hhll hh, (v,u) <- ssll ss]
    size = fromInteger . historiesSize
    hhll = historiesList
    vars = historiesVars
    ssll = statesList
    uvals uu v = fromJust $ systemsVarsSetValue uu v
    uval uu v = card $ fromJust $ systemsVarsSetValue uu v
    uull = systemsList
    llrr = R.fromListUnboxed
    llmm :: forall k a. Ord k => [(k, a)] -> Map.Map k a
    llmm = Map.fromList
    card = Set.size
    qqll = Set.toList
    llvv = V.fromList
    llvu = UV.fromList

-- AYOR
systemsHistoryRepasHistory_u :: System -> HistoryRepa -> Maybe History
systemsHistoryRepasHistory_u uu aa
  | vvqq vv `subset` uvars uu = 
      llhh [(IdInt (toInteger (i+1)), llss [(v,w) | j <- [0..n-1], let v = vv V.! j, 
        let k = rr R.! (Z :. j :. i), let w = mm Map.! v V.! (fromIntegral k)]) | i <- [0..z-1]]
  | otherwise = Nothing
  where 
    vv = historyRepasVectorVar aa
    rr = historyRepasArray aa
    mm = llmm [(v, llvv (qqll ww)) | v <- V.toList vv, let ww = uvals uu v]
    Z :. n :. z = extent rr
    llhh = listsHistory
    llss = listsState
    uvals uu v = fromJust $ systemsVarsSetValue uu v
    uvars = systemsVars
    uull = systemsList
    llmm = Map.fromList
    qqll = Set.toList
    vvqq = Set.fromList . V.toList
    subset = Set.isSubsetOf
    llvv = V.fromList
    vvll = V.toList

systemsTransformsTransformRepa :: System -> Transform -> Maybe TransformRepa
systemsTransformsTransformRepa uu tt
  | tt /= empty && isOneFunc uu tt && card (der tt) == 1 = 
      Just $ TransformRepa (llvv vv) mvv w (fromIntegral (uval uu w)) (llrr sh (elems nn))
  | otherwise = Nothing
  where 
    vv = qqll $ und tt
    mvv = llmm (zip vv [0..])
    mm = llmm [(v, llmm (zip (qqll ww) [0..])) | v <- w:vv, let ww = uvals uu v]
    sh = shapeOfList [Map.size (mm Map.! v) | v <- vv] :: VShape
    w = Set.findMin (der tt)
    nn = llim [(toIndex sh (shapeOfList [mm Map.! v Map.! (ss `sat` v) | v <- vv] :: VShape), 
                fromIntegral (mm Map.! w Map.! (ss `sat` w))) | (ss,_) <- aall (ttaa tt)]
    isOneFunc = systemsTransformsIsOneFunc
    empty = transformEmpty
    und = transformsUnderlying
    der = transformsDerived
    ttaa = transformsHistogram
    aall = histogramsList
    sat ss v = fromJust $ statesVarsValue ss v
    uvals uu v = fromJust $ systemsVarsSetValue uu v
    uval uu v = card $ fromJust $ systemsVarsSetValue uu v
    uull = systemsList
    llrr = R.fromListUnboxed
    elems = IntMap.elems
    llim = IntMap.fromList
    llmm :: forall k a. Ord k => [(k, a)] -> Map.Map k a
    llmm = Map.fromList
    qqll = Set.toList
    card = Set.size
    llvv = V.fromList

systemsTransformsTransformRepa_u :: System -> Transform -> TransformRepa
systemsTransformsTransformRepa_u uu tt = TransformRepa (llvv vv) mvv w (fromIntegral (uval uu w)) (llrr sh (elems nn))
  where 
    vv = qqll $ und tt
    mvv = llmm (zip vv [0..])
    mm = llmm [(v, llmm (zip (qqll ww) [0..])) | v <- w:vv, let ww = uvals uu v]
    sh = shapeOfList [Map.size (mm Map.! v) | v <- vv] :: VShape
    w = Set.findMin (der tt)
    nn = llim [(toIndex sh (shapeOfList [mm Map.! v Map.! (ss `sat` v) | v <- vv] :: VShape), 
                fromIntegral (mm Map.! w Map.! (ss `sat` w))) | (ss,_) <- aall (ttaa tt)]
    und = transformsUnderlying
    der = transformsDerived
    ttaa = transformsHistogram
    aall = histogramsList
    sat ss v = fromJust $ statesVarsValue ss v
    uvals uu v = fromJust $ systemsVarsSetValue uu v
    uval uu v = card $ uvals uu v
    uull = systemsList
    llrr = R.fromListUnboxed
    elems = IntMap.elems
    llim = IntMap.fromList
    llmm :: forall k a. Ord k => [(k, a)] -> Map.Map k a
    llmm = Map.fromList
    qqll = Set.toList
    card = Set.size
    llvv = V.fromList

histogramRepaVecsSum :: HistogramRepaVec -> UV.Vector Double
histogramRepaVecsSum (HistogramRepaVec _ _ _ _ vaa) = V.convert (V.map UV.sum vaa)

histogramRepaVecsFaclnsRepaVecs :: HistogramRepaVec -> HistogramRepaVec
histogramRepaVecsFaclnsRepaVecs (HistogramRepaVec vvv mvv _ svv !vaa) = 
    HistogramRepaVec vvv mvv 1 svv (V.map (UV.map (\x -> logGamma (x + 1))) vaa)

setSetVarsHistogramRepasPartition_u :: Set.Set (Set.Set Variable) -> HistogramRepa -> HistogramRepa 
setSetVarsHistogramRepasPartition_u pp aa = rraa rr'
  where
    HistogramRepa vvv mvv !rr = aa
    !svv = extent rr
    !n = rank svv
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !spp = V.map (\pcc -> perm svv pcc) ppp 
    !sxx = V.convert $ V.map R.size spp
    !pvv = llvu $ [mvv Map.! v | cc <- qqll pp, v <- qqll cc] 
    !pww = llvu $ snd $ unzip $ sort $ zip (vull pvv) [0..]
    rr' = R.computeS $ R.reshape sxx $ R.backpermute (perm svv pvv) (\iww -> perm iww pww) rr
    rraa = arraysHistogramRepa
    perm = UV.unsafeBackpermute
    vull = UV.toList
    llvu = UV.fromList
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList

setSetVarsHistogramRepaVecsPartitionVec_u_1 :: Set.Set (Set.Set Variable) -> HistogramRepaVec -> HistogramRepaVec 
setSetVarsHistogramRepaVecsPartitionVec_u_1 pp rrv = HistogramRepaVec vyy myy z syy vbb
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !p = V.length vaa
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !m = V.length vpp
    yy = List.map VarIndex [0 .. m-1]
    vyy = llvv yy
    myy = llmm (zip yy [0..])
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !spp = V.map (\pcc -> perm svv pcc) ppp 
    !syy = V.convert $ V.map R.size spp
    !pvv = llvu $ [mvv Map.! v | cc <- qqll pp, v <- qqll cc] 
    !sww = perm svv pvv
    !vbb = runST $ do
      vbb <- V.replicateM p (MV.replicate v 0)
      rvv <- newSTRef (UV.replicate n 0)
      forM_ [0 .. v-1] $ (\j -> do 
        ivv <- readSTRef rvv
        let !i = R.toIndex sww $ perm ivv pvv
        forM_ [0 .. p-1] $ (\l -> do 
          MV.unsafeWrite (vbb V.! l) i (vaa V.! l UV.! j))
        writeSTRef rvv (incIndex svv ivv))
      V.mapM UV.unsafeFreeze vbb
    llmm = Map.fromList
    perm = UV.unsafeBackpermute
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llvu = UV.fromList

setSetVarsHistogramRepaVecsPartitionVec_u :: Set.Set (Set.Set Variable) -> HistogramRepaVec -> HistogramRepaVec 
setSetVarsHistogramRepaVecsPartitionVec_u !pp !rrv = HistogramRepaVec vyy myy z syy vbb
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !p = V.length vaa
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !m = V.length vpp
    yy = List.map VarIndex [0 .. m-1]
    vyy = llvv yy
    myy = llmm (zip yy [0..])
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !spp = V.map (\pcc -> perm svv pcc) ppp 
    !syy = V.convert $ V.map R.size spp
    !pvv = llvu $ [mvv Map.! v | cc <- qqll pp, v <- qqll cc] 
    !sww = perm svv pvv
    !vbb = runST $ do
      vbb <- V.replicateM p (MV.replicate v 0)
      !ivv <- MV.replicate n 0
      forM_ [0 .. v-1] $ (\j -> do 
        !i <- toIndexPermM sww pvv ivv
        forM_ [0 .. p-1] $ (\l -> do 
          MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex (V.unsafeIndex vaa l) j))
        incIndexM_ svv ivv)
      V.mapM UV.unsafeFreeze vbb
    llmm = Map.fromList
    perm = UV.unsafeBackpermute
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llvu = UV.fromList

sumsHistogramRepa4VecsRollMapPairsHistogramRepa4VecsSum_u :: Double -> HistogramRepaVec -> 
  (UV.Vector Int, UV.Vector Int) -> HistogramRepaVec -> UV.Vector Double
sumsHistogramRepa4VecsRollMapPairsHistogramRepa4VecsSum_u !a !aav (!rs,!rt) !rrv = bb
  where
    HistogramRepaVec _ _ _ svv vaa = aav
    HistogramRepaVec _ _ _ syy vrr = rrv
    !d = UV.unsafeIndex svv 0
    !r = UV.unsafeIndex syy 0
    [!a1,!a2,!b1,!b2] = V.toList vaa 
    [!ra1,!ra2,!rb1,!rb2] = V.toList vrr 
    !bb = UV.create $ do
      !bb <- MV.replicate r 0
      forM_ [0 .. r-1] $ (\j -> do
        let !s = UV.unsafeIndex rs j  
        let !t = UV.unsafeIndex rt j  
        MV.unsafeWrite bb j (a 
          + UV.unsafeIndex ra1 j - UV.unsafeIndex a1 s - UV.unsafeIndex a1 t
          - UV.unsafeIndex ra2 j + UV.unsafeIndex a2 s + UV.unsafeIndex a2 t
          - UV.unsafeIndex rb1 j + UV.unsafeIndex b1 s + UV.unsafeIndex b1 t
          + UV.unsafeIndex rb2 j - UV.unsafeIndex b2 s - UV.unsafeIndex b2 t))
      return bb

sumsHistogramRepa4VecsRollMapPairsHistogramRepa4VecsSum_u_1 :: UV.Vector Double -> HistogramRepaVec -> 
  (UV.Vector Int, UV.Vector Int) -> HistogramRepaVec -> UV.Vector Double
sumsHistogramRepa4VecsRollMapPairsHistogramRepa4VecsSum_u_1 !av !aav (!rs,!rt) !rrv = bb
  where
    HistogramRepaVec _ _ _ _ vaa = aav
    HistogramRepaVec _ _ _ _ vrr = rrv
    [!f1,!f2,!g1,!g2] = UV.toList av 
    [!a1,!a2,!b1,!b2] = V.toList vaa 
    [!ra1,!ra2,!rb1,!rb2] = V.toList vrr 
    !bb = (UV.imap (\i x -> x - a1 UV.! (rs UV.! i) - a1 UV.! (rt UV.! i) + f1) ra1 `sub`
           UV.imap (\i x -> x - a2 UV.! (rs UV.! i) - a2 UV.! (rt UV.! i) + f2) ra2 `sub`
           UV.imap (\i x -> x - b1 UV.! (rs UV.! i) - b1 UV.! (rt UV.! i) + g1) rb1 `add`
           UV.imap (\i x -> x - b2 UV.! (rs UV.! i) - b2 UV.! (rt UV.! i) + g2) rb2)
    add = UV.zipWith (+)
    sub = UV.zipWith (-)

sumsHistogramRepasRollMapPairsHistogramRepasSum_u :: Double -> HistogramRepa -> 
  (UV.Vector Int, UV.Vector Int) -> HistogramRepa -> UV.Vector Double
sumsHistogramRepasRollMapPairsHistogramRepasSum_u !a !aav (!rs,!rt) !rrv = bb
  where
    HistogramRepa _ _  aa = aav
    HistogramRepa _ _  rr = rrv
    !syy = extent rr
    !aa' = R.toUnboxed aa
    !rr' = R.toUnboxed rr
    !r = UV.unsafeIndex syy 0
    !bb = UV.create $ do
      !bb <- MV.replicate r 0
      forM_ [0 .. r-1] $ (\j -> do
        let !s = UV.unsafeIndex rs j  
        let !t = UV.unsafeIndex rt j  
        MV.unsafeWrite bb j (a + UV.unsafeIndex rr' j - UV.unsafeIndex aa' s - UV.unsafeIndex aa' t))
      return bb

setVarsHistogramRepasReduce :: Set.Set Variable -> HistogramRepa -> HistogramRepa 
setVarsHistogramRepasReduce kk aa 
  | V.null vjj = aa
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [sumAllS rr])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    HistogramRepa vvv mvv !rr = aa
    !vv = llqq $ vvll vvv
    !svv = extent rr
    !n = rank svv
    !vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    !vjj = llvv $ qqll (vv `minus` kk)
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !rr' = R.fromUnboxed skk $ UV.create $ do
      !mv <- MV.replicate (R.size skk) 0
      !ivv <- MV.replicate n 0
      UV.forM_ (toUnboxed rr) $ \a -> do 
        !i <- toIndexPermM skk pkk ivv
        !c <- MV.unsafeRead mv i
        MV.unsafeWrite mv i (c+a)
        incIndexM_ svv ivv
      return mv
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    minus = Set.difference
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList

setVarsHistogramRepasReduce_1 :: Set.Set Variable -> HistogramRepa -> HistogramRepa 
setVarsHistogramRepasReduce_1 kk aa 
  | V.null vjj = aa
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [sumAllS rr])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    vvv = histogramRepasVectorVar aa
    vv = llqq $ vvll vvv
    rr = histogramRepasArray aa
    mvv = histogramRepasMapVarInt aa
    svv = extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    vjj = llvv $ qqll (vv `minus` kk)
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !pjj = llvu $ vvll $ V.map (mvv Map.!) vjj
    !qvv = llvu $ snd $ unzip $ sort $ zip (vull $ pkk UV.++ pjj) [0..]
    !skk = perm svv pkk
    !sjj = perm svv pjj
    !syy = skk R.:. UV.foldl' (*) 1 (perm svv pjj) 
    rr' = sumS $ R.backpermute syy back rr
    back !(ikk R.:. i) = perm (ikk UV.++ ijj) qvv
      where
         !ijj = R.fromIndex sjj i
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    minus = Set.difference
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList
    vull = UV.toList

setVarsHistogramRepasReduce_2 :: Set.Set Variable -> HistogramRepa -> HistogramRepa 
setVarsHistogramRepasReduce_2 kk aa 
  | V.null vjj = aa
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [sumAllS rr])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    HistogramRepa vvv mvv !rr = aa
    vv = llqq $ vvll vvv
    !svv = extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    vjj = llvv $ qqll (vv `minus` kk)
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !rr' = R.fromUnboxed skk $ UV.create $ do
      mv <- MV.replicate (R.size skk) 0
      mapM_ (\(i,a) -> do c <- MV.read mv i; MV.write mv i (c+a)) 
            [(R.toIndex skk (perm ivv pkk),a) | 
               (!j,!a) <- zip [0..] (UV.toList (toUnboxed rr)), let !ivv = R.fromIndex svv j]
      return mv
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    minus = Set.difference
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList

setVarsHistogramRepasReduce_3 :: Set.Set Variable -> HistogramRepa -> HistogramRepa 
setVarsHistogramRepasReduce_3 kk aa 
  | V.null vjj = aa
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [sumAllS rr])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    HistogramRepa vvv mvv !rr = aa
    vv = llqq $ vvll vvv
    !svv = extent rr
    !n = rank svv
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    vjj = llvv $ qqll (vv `minus` kk)
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !rr' = R.fromUnboxed skk $ UV.create $ do
      mv <- MV.replicate (R.size skk) 0
      mapM_ (\(i,a) -> do c <- MV.read mv i; MV.write mv i (c+a)) 
            [(R.toIndex skk (perm ivv pkk),a) | 
               (!ivv,!a) <- zip (linc svv (UV.replicate n 0)) (UV.toList (toUnboxed rr))]
      return mv
    linc !svv !ivv = let !jvv = incIndex svv ivv in ivv : linc svv jvv 
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    minus = Set.difference
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList

setVarsHistogramRepasReduce_4 :: Set.Set Variable -> HistogramRepa -> HistogramRepa 
setVarsHistogramRepasReduce_4 kk aa 
  | V.null vjj = aa
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [sumAllS rr])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    HistogramRepa vvv mvv !rr = aa
    !vv = llqq $ vvll vvv
    !svv = extent rr
    !n = rank svv
    !vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    !vjj = llvv $ qqll (vv `minus` kk)
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !rr' = R.fromUnboxed skk $ UV.create $ do
      !mv <- MV.replicate (R.size skk) 0
      !xvv <- newSTRef (UV.replicate n 0)
      UV.forM_ (toUnboxed rr) $ \a -> do 
        !ivv <- readSTRef xvv
        let !i = R.toIndex skk (perm ivv pkk)
        !c <- MV.read mv i
        MV.write mv i (c+a)
        writeSTRef xvv (incIndex svv ivv)
      return mv
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    minus = Set.difference
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList

varsHistogramRepaVecsReduceSingleVec_u :: Int -> HistogramRepaVec -> HistogramRepaVec 
varsHistogramRepaVecsReduceSingleVec_u !u !rrv = HistogramRepaVec vyy myy z syy vbb
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !d = UV.unsafeIndex svv u
    !x = V.unsafeIndex vvv u
    vyy = V.singleton x
    myy = Map.singleton x 0
    !syy = UV.singleton d
    !pvv = UV.singleton u 
    !p = V.length vaa
    !vbb = runST $ do
      vbb <- V.replicateM p (MV.replicate d 0)
      !ivv <- MV.replicate n 0
      forM_ [0 .. v-1] $ (\j -> do 
        !i <- toIndexPermM syy pvv ivv
        forM_ [0 .. p-1] $ (\l -> do 
          let !mv = V.unsafeIndex vbb l
          c <- MV.unsafeRead mv i
          let !a = UV.unsafeIndex (V.unsafeIndex vaa l) j
          MV.unsafeWrite mv i (c+a))
        incIndexM_ svv ivv)
      V.mapM UV.unsafeFreeze vbb

varsHistogramRepa4VecsReduceSingle_u :: Int -> HistogramRepaVec -> HistogramRepa
varsHistogramRepa4VecsReduceSingle_u !u !rrv = HistogramRepa vyy myy bb
  where
    HistogramRepaVec vvv mvv _ svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !d = UV.unsafeIndex svv u
    !x = V.unsafeIndex vvv u
    vyy = V.singleton x
    myy = Map.singleton x 0
    !syy = UV.singleton d
    !va1 = V.unsafeIndex vaa 0
    !va2 = V.unsafeIndex vaa 1
    !vb1 = V.unsafeIndex vaa 2
    !vb2 = V.unsafeIndex vaa 3
    !bb = R.fromUnboxed syy $ UV.create $ do
      !bb <- MV.replicate d 0
      !ivv <- MV.replicate n 0
      forM_ [0 .. v-1] $ (\j -> do 
        i <- MV.unsafeRead ivv u
        c <- MV.unsafeRead bb i
        let !a1 = UV.unsafeIndex va1 j
        let !a2 = UV.unsafeIndex va2 j
        let !b1 = UV.unsafeIndex vb1 j
        let !b2 = UV.unsafeIndex vb2 j
        MV.unsafeWrite bb i (c+a1-a2-b1+b2)
        incIndexM_ svv ivv)
      return bb

histogramRepasRed_u :: Double -> HistogramRepa -> HistogramRepaRed 
histogramRepasRed_u z aa = HistogramRepaRed vvv mvv svv lrr
  where
    HistogramRepa vvv mvv !rr = aa
    !svv = extent rr
    !n = rank svv
    !f = 1 / z
    !lrr = runST $ do
      vrr <- V.generateM n (\i -> MV.replicate (svv UV.! i) 0)
      mapM_ (\(mw,i,a) -> do c <- MV.read mw i; MV.write mw i (c+a)) 
            [(mw,(ivv UV.! k),a) | (!j,!a) <- zip [0..] (UV.toList (toUnboxed rr)), 
                                  let !ivv = R.fromIndex svv j, !k <- [0..n-1], let !mw = vrr V.! k]
      mapM_ (\(mw,i) -> do c <- MV.read mw i; MV.write mw i (c*f)) 
            [(mw,i) | f /= 1, !k <- [0..n-1], let !mw = vrr V.! k, !i <- [0.. (svv UV.! k)-1]]
      V.mapM UV.unsafeFreeze vrr

histogramRepaVecsRedVec :: HistogramRepaVec -> HistogramRepaRedVec 
histogramRepaVecsRedVec !rrv = HistogramRepaRedVec vvv mvv z svv vxx
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !p = V.length vaa
    !vxx = runST $ do
      vxx <- V.replicateM p (V.generateM n (\i -> MV.replicate (UV.unsafeIndex svv i) 0))
      !ivv <- MV.replicate n 0
      forM_ [0 .. v-1] $ (\j -> do 
        forM_ [0 .. n-1] $ (\k -> do 
          !i <- MV.unsafeRead ivv k
          forM_ [0 .. p-1] $ (\l -> do 
            let !mv = V.unsafeIndex (V.unsafeIndex vxx l) k
            c <- MV.unsafeRead mv i
            let !a = UV.unsafeIndex (V.unsafeIndex vaa l) j
            MV.unsafeWrite mv i (c+a)))
        incIndexM_ svv ivv)
      V.mapM (V.mapM UV.unsafeFreeze) vxx

histogramRepa4VecsRed_u :: HistogramRepaVec -> HistogramRepaRed
histogramRepa4VecsRed_u !aav = HistogramRepaRed vvv mvv svv vbb
  where
    HistogramRepaVec vvv mvv _ svv vaa = aav
    !v = R.size svv
    !n = rank svv
    !va1 = V.unsafeIndex vaa 0
    !va2 = V.unsafeIndex vaa 1
    !vb1 = V.unsafeIndex vaa 2
    !vb2 = V.unsafeIndex vaa 3
    !vbb = runST $ do
      vbb <- V.generateM n (\i -> MV.replicate (UV.unsafeIndex svv i) 0)
      !ivv <- MV.replicate n 0
      forM_ [0 .. v-1] $ (\j -> do 
        let !a1 = UV.unsafeIndex va1 j
        let !a2 = UV.unsafeIndex va2 j
        let !b1 = UV.unsafeIndex vb1 j
        let !b2 = UV.unsafeIndex vb2 j
        forM_ [0 .. n-1] $ (\k -> do 
          i <- MV.unsafeRead ivv k
          let !bb = V.unsafeIndex vbb k
          c <- MV.unsafeRead bb i
          MV.unsafeWrite bb i (c+a1-a2-b1+b2))
        incIndexM_ svv ivv)
      V.mapM UV.unsafeFreeze vbb

varsHistogramRepaRedVecsSingleVec_u :: Int -> HistogramRepaRedVec -> HistogramRepaVec 
varsHistogramRepaRedVecsSingleVec_u !u !xxv = HistogramRepaVec vyy myy z syy vbb
  where
    HistogramRepaRedVec vvv mvv z svv vxx = xxv
    !d = UV.unsafeIndex svv u
    !x = V.unsafeIndex vvv u
    vyy = V.singleton x
    myy = Map.singleton x 0
    !syy = UV.singleton d
    !vbb = V.map (\xx -> V.unsafeIndex xx u) vxx

varsHistogramRepaRedsSingle_u :: Int -> HistogramRepaRed -> HistogramRepa
varsHistogramRepaRedsSingle_u !u !xxv = HistogramRepa vyy myy bb
  where
    HistogramRepaRed vvv mvv svv vxx = xxv
    !d = UV.unsafeIndex svv u
    !x = V.unsafeIndex vvv u
    vyy = V.singleton x
    myy = Map.singleton x 0
    !syy = UV.singleton d
    !bb = R.fromUnboxed syy $ V.unsafeIndex vxx u

setSetVarsHistogramRepasPartitionRed_u :: Double -> Set.Set (Set.Set Variable) -> HistogramRepa -> HistogramRepaRed 
setSetVarsHistogramRepasPartitionRed_u z pp aa = HistogramRepaRed vxx mxx sxx lrr
  where
    HistogramRepa vvv mvv !rr = aa
    !svv = extent rr
    !n = rank svv
    !f = 1 / z
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !m = V.length vpp
    xx = List.map VarIndex [0 .. m-1]
    vxx = llvv xx
    mxx = llmm (zip xx [0..])
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !spp = V.map (\pcc -> perm svv pcc) ppp 
    !sxx = V.convert $ V.map R.size spp
    !lrr = runST $ do
      vrr <- V.generateM m (\i -> MV.replicate (sxx UV.! i) 0)
      mapM_ (\(mw,i,a) -> do c <- MV.read mw i; MV.write mw i (c+a)) 
            [(vrr V.! k, (R.toIndex (spp V.! k) $ perm ivv (ppp V.! k)), a) | 
              (!j,!a) <- zip [0..] (UV.toList (toUnboxed rr)), let !ivv = R.fromIndex svv j, !k <- [0..m-1]]
      mapM_ (\(mw,i) -> do c <- MV.read mw i; MV.write mw i (c*f)) 
            [(vrr V.! k, i) | f /= 1, !k <- [0..m-1], !i <- [0.. (sxx UV.! k)-1]]
      V.mapM UV.unsafeFreeze vrr
    llmm = Map.fromList
    perm = UV.unsafeBackpermute
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList

setSetVarsHistogramRepasPartitionRed_u_1 :: Double -> Set.Set (Set.Set Variable) -> HistogramRepa -> HistogramRepaRed 
setSetVarsHistogramRepasPartitionRed_u_1 z pp aa = aaax z $ rraa rr'
  where
    HistogramRepa vvv mvv !rr = aa
    !svv = extent rr
    !n = rank svv
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !spp = V.map (\pcc -> perm svv pcc) ppp 
    !sxx = V.convert $ V.map R.size spp
    !pvv = llvu $ [mvv Map.! v | cc <- qqll pp, v <- qqll cc] 
    !pww = llvu $ snd $ unzip $ sort $ zip (vull pvv) [0..]
    rr' = R.computeS $ R.reshape sxx $ R.backpermute (perm svv pvv) (\iww -> perm iww pww) rr
    aaax = histogramRepasRed_u
    rraa = arraysHistogramRepa
    perm = UV.unsafeBackpermute
    vull = UV.toList
    llvu = UV.fromList
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList

setSetVarsHistogramRepaVecsPartitionRedVec_u :: Set.Set (Set.Set Variable) -> HistogramRepaVec -> HistogramRepaRedVec 
setSetVarsHistogramRepaVecsPartitionRedVec_u pp rrv = HistogramRepaRedVec vyy myy z syy vxx
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !p = V.length vaa
    !f = 1 / z
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !m = V.length vpp
    yy = List.map VarIndex [0 .. m-1]
    vyy = llvv yy
    myy = llmm (zip yy [0..])
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !spp = V.map (\pcc -> perm svv pcc) ppp 
    !syy = V.convert $ V.map R.size spp
    !vxx = runST $ do
      vxx <- V.replicateM p (V.generateM m (\i -> MV.replicate (UV.unsafeIndex syy i) 0))
      !ivv <- MV.replicate n 0
      forM_ [0 .. v-1] $ (\j -> do 
        forM_ [0 .. m-1] $ (\k -> do 
          !i <- toIndexPermM (V.unsafeIndex spp k) (V.unsafeIndex ppp k) ivv
          forM_ [0 .. p-1] $ (\l -> do 
            let !mv = V.unsafeIndex (V.unsafeIndex vxx l) k
            c <- MV.unsafeRead mv i
            let !a = UV.unsafeIndex (V.unsafeIndex vaa l) j
            MV.unsafeWrite mv i (c+a)))
        incIndexM_ svv ivv)
      if f /= 1 
        then do
          forM_ [0 .. p-1] $ (\l -> do 
            forM_ [0 .. m-1] $ (\k -> do 
              let !mv = V.unsafeIndex (V.unsafeIndex vxx l) k
              forM_ [0..(MV.length mv)-1] $ (\i -> do 
                c <- MV.unsafeRead mv i
                MV.unsafeWrite mv i (c*f))))
        else do
          return ()
      V.mapM (V.mapM UV.unsafeFreeze) vxx
    llmm = Map.fromList
    perm = UV.unsafeBackpermute
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList

setSetVarsHistogramRepaVecsPartitionRedVec_u_1 :: Set.Set (Set.Set Variable) -> HistogramRepaVec -> HistogramRepaRedVec 
setSetVarsHistogramRepaVecsPartitionRedVec_u_1 pp rrv = HistogramRepaRedVec vyy myy z syy vxx
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !p = V.length vaa
    !f = 1 / z
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !m = V.length vpp
    yy = List.map VarIndex [0 .. m-1]
    vyy = llvv yy
    myy = llmm (zip yy [0..])
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !spp = V.map (\pcc -> perm svv pcc) ppp 
    !syy = V.convert $ V.map R.size spp
    !vxx = runST $ do
      vxx <- V.replicateM p (V.generateM m (\i -> MV.replicate (syy UV.! i) 0))
      rvv <- newSTRef (UV.replicate n 0)
      forM_ [0 .. v-1] $ (\j -> do 
        ivv <- readSTRef rvv
        forM_ [0 .. m-1] $ (\k -> do 
          let !i = R.toIndex (spp V.! k) $ perm ivv (ppp V.! k)
          forM_ [0 .. p-1] $ (\l -> do 
            let !mv = vxx V.! l V.! k
            c <- MV.unsafeRead mv i
            MV.unsafeWrite mv i (c + (vaa V.! l UV.! j))))
        writeSTRef rvv (incIndex svv ivv))
      mapM_ (\(mv,i) -> do c <- MV.read mv i; MV.write mv i (c*f)) 
            [(vxx V.! l V.! k, i) | f /= 1, !l <- [0..p-1], !k <- [0..m-1], !i <- [0..(syy UV.! k)-1]]
      V.mapM (V.mapM UV.unsafeFreeze) vxx
    llmm = Map.fromList
    perm = UV.unsafeBackpermute
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList

setVarsHistoryRepasCountApproxs :: Set.Set Variable -> HistoryRepa -> [Int] 
setVarsHistoryRepasCountApproxs kk hh 
  | V.null vkk = [z]
  | otherwise = IntMap.elems ll
  where
    HistoryRepa vvv mvv svv !rr = hh
    vv = llqq $ vvll vvv
    Z :. _ :. (!z) = extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !rr' = UV.generate z (\j -> UV.foldl' (\a i -> a * 23 + (rr R.! (Z :. i :. j))) 0 pkk)
    ll = IntMap.fromListWith (+) [(fromIntegral i,1) | i <- vull rr']
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList
    vull = UV.toList

setVarsHistoryRepasReduce :: Double -> Set.Set Variable -> HistoryRepa -> HistogramRepa 
setVarsHistoryRepasReduce f kk hh
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [fromIntegral z])
  | otherwise = setVarsHistoryRepaStorablesReduce f kk hh vsh
  where
    HistoryRepa vvv mvv svv rr = hh
    vsh = SV.unsafeCast (UV.convert (R.toUnboxed rr)) :: SV.Vector CShort
    vv = llqq $ vvll vvv
    Z :. _ :. z = extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList

foreign import ccall unsafe "listVarsArrayHistoriesReduce_u" listVarsArrayHistoriesReduce_u
    :: CDouble -> CLLong -> Ptr CLLong -> Ptr CLLong -> CLLong -> Ptr CShort -> Ptr CDouble -> IO ()

setVarsHistoryRepaStorablesReduce :: Double -> Set.Set Variable -> HistoryRepa -> SV.Vector CShort -> HistogramRepa 
setVarsHistoryRepaStorablesReduce !f !kk !hh !vsh
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [fromIntegral z])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    HistoryRepa vvv mvv svv !rr = hh
    vv = llqq $ vvll vvv
    R.Z R.:. _ R.:. (!z) = R.extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !vspkk = SV.unsafeCast (UV.convert pkk) :: SV.Vector CLLong
    !skk = perm svv pkk
    !vsskk = SV.unsafeCast (UV.convert skk) :: SV.Vector CLLong
    !m = UV.length pkk
    !rr' = R.fromUnboxed skk $ SV.convert $ SV.unsafeCast $ unsafePerformIO $ do
      let vs = SV.replicate (R.size skk) 0
      mv <- SV.unsafeThaw vs
      SV.unsafeWith vspkk $ \ppkk -> do
      SV.unsafeWith vsskk $ \pskk -> do
      SV.unsafeWith vsh $ \prr -> do
      SMV.unsafeWith mv $ \pmv -> do
        listVarsArrayHistoriesReduce_u (realToFrac f) (fromIntegral m) ppkk pskk (fromIntegral z) prr pmv
      SV.unsafeFreeze mv 
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList

setVarsHistoryRepasReduce_1 :: Double -> Set.Set Variable -> HistoryRepa -> HistogramRepa 
setVarsHistoryRepasReduce_1 f kk hh 
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [fromIntegral z])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    HistoryRepa vvv mvv svv !rr = hh
    vv = llqq $ vvll vvv
    Z :. _ :. (!z) = extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !rr' = R.fromUnboxed skk $ UV.create $ do
      mv <- MV.replicate (R.size skk) 0
      mapM_ (\i -> do c <- MV.unsafeRead mv i; MV.unsafeWrite mv i (c+f)) 
            [R.toIndex skk (UV.map (\i -> fromIntegral (rr R.! (Z :. i :. j))) pkk) | !j <- [0 .. z-1]]
      return mv
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList

setVarsHistoryRepasReduce_2 :: Double -> Set.Set Variable -> HistoryRepa -> HistogramRepa 
setVarsHistoryRepasReduce_2 f kk hh 
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [fromIntegral z])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    HistoryRepa vvv mvv svv !rr = hh
    vv = llqq $ vvll vvv
    Z :. _ :. (!z) = extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !rr' = R.fromUnboxed skk $ UV.create $ do
      mv <- MV.replicate (R.size skk) 0
      mapM_ (\i -> do c <- MV.unsafeRead mv i; MV.unsafeWrite mv i (c+f)) 
            [UV.ifoldl' (\a k i -> a * (skk UV.! k) + (fromIntegral (rr R.! (Z :. i :. j)))) 0 pkk | !j <- [0 .. z-1]]
      return mv
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList

setVarsHistoryRepasReduce_3 :: Double -> Set.Set Variable -> HistoryRepa -> HistogramRepa 
setVarsHistoryRepasReduce_3 !f kk hh 
  | V.null vkk = HistogramRepa vempty mempty (llrr vsempty [fromIntegral z])
  | otherwise = HistogramRepa vkk mkk rr'
  where
    HistoryRepa vvv mvv svv !rr = hh
    vv = llqq $ vvll vvv
    Z :. _ :. (!z) = extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !rr' = R.fromUnboxed skk $ UV.create $ do
      mv <- MV.replicate (R.size skk) 0
      loop mv 0
    loop mv !j = do
      if j==z then
        return mv
      else do
        let !p = UV.ifoldl' (\a k i -> a * (UV.unsafeIndex skk k) + (fromIntegral (R.unsafeIndex rr (Z :. i :. j)))) 0 pkk
        c <- MV.unsafeRead mv p
        MV.unsafeWrite mv p (c+f)
        loop mv (j+1)
    vsempty = UV.empty
    llrr = R.fromListUnboxed
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    vempty = V.empty
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList

historyRepasRed :: HistoryRepa -> HistogramRepaRed 
historyRepasRed hh = HistogramRepaRed vvv mvv svv lrr
  where
    HistoryRepa vvv mvv svv !rr = hh
    Z :. n :. (!z) = extent rr
    !f = 1 / fromIntegral z
    !lrr = llvv n $ [(UV.create $ do
      mv <- MV.replicate (svv UV.! i) 0
      mapM_ (\k -> do c <- MV.read mv k; MV.write mv k (c+f)) [fromIntegral (rr R.! (Z :. i :. j)) | !j <- [0 .. z-1]]
      return mv) 
      | i <- [0 .. n-1]]
    llvv = V.fromListN

setVarsHistoryRepasRed :: Set.Set Variable -> HistoryRepa -> HistogramRepaRed 
setVarsHistoryRepasRed kk hh 
  | V.null vkk = HistogramRepaRed vempty mempty vsempty vempty
  | otherwise = HistogramRepaRed vkk mkk skk lrr
  where
    HistoryRepa vvv mvv svv !rr = hh
    vv = llqq $ vvll vvv
    Z :. n :. (!z) = extent rr
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    !m = V.length vkk
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !f = 1 / fromIntegral z
    !lrr = llvv $ [(UV.create $ do
      mv <- MV.replicate (skk UV.! i) 0
      mapM_ (\k -> do c <- MV.read mv k; MV.write mv k (c+f)) [fromIntegral (rr R.! (Z :. p :. j)) | !j <- [0 .. z-1]]
      return mv) 
      | !i <- [0 .. m-1], let !p = pkk UV.! i]
    perm = UV.unsafeBackpermute
    vsempty = UV.empty
    mempty = Map.empty
    llmm = Map.fromList
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    vempty = V.empty
    vvll = V.toList
    llvv = V.fromList
    llvu = UV.fromList

setVarsHistogramRepaRedsRed :: Set.Set Variable -> HistogramRepaRed -> HistogramRepaRed 
setVarsHistogramRepaRedsRed kk hh 
  | V.null vkk = HistogramRepaRed vempty mempty vsempty vempty
  | otherwise = HistogramRepaRed vkk mkk (vuperm svv pukk) (vperm lrr pkk)
  where
    HistogramRepaRed vvv mvv svv lrr = hh
    vv = llqq $ vvll vvv
    vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    pukk = llvu $ vvll $ V.map (mvv Map.!) vkk
    pkk = V.map (mvv Map.!) vkk
    vuperm = UV.unsafeBackpermute
    vperm = V.unsafeBackpermute
    vsempty = UV.empty
    llmm = Map.fromList
    mempty = Map.empty
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    vempty = V.empty
    vvll = V.toList
    llvv = V.fromList
    llvu = UV.fromList

histogramRepaRedsIndependent :: Double -> HistogramRepaRed -> HistogramRepa 
histogramRepaRedsIndependent z hh 
  | V.null vvv = HistogramRepa vempty mempty (llrr vsempty [z])
  | otherwise = HistogramRepa vvv mvv rr'
  where
    HistogramRepaRed vvv mvv svv !lrr = hh
    !v = R.size svv
    !n = R.rank svv
    !rr' = R.fromUnboxed svv $ UV.create $ do
      mv <- MV.replicate v 0
      xvv <- newSTRef (UV.replicate n 0)
      forM_ [0 .. v-1] $ \j -> do 
        ivv <- readSTRef xvv
        MV.unsafeWrite mv j (z * UV.foldl1' (*) (UV.imap (\p i -> lrr V.! p UV.! i) ivv))
        writeSTRef xvv (incIndex svv ivv)
      return mv
    mempty = Map.empty
    vempty = V.empty
    llrr = R.fromListUnboxed
    vsempty = UV.empty
    vvll = V.toList
 
histogramRepaRedsIndependent_1 :: Double -> HistogramRepaRed -> HistogramRepa 
histogramRepaRedsIndependent_1 z hh 
  | V.null vvv = HistogramRepa vempty mempty (llrr vsempty [z])
  | otherwise = HistogramRepa vvv mvv rr'
  where
    HistogramRepaRed vvv mvv svv !lrr = hh
    !v = R.size svv
    !n = R.rank svv
    !rr' = R.fromUnboxed svv $ UV.create $ do
      mv <- MV.replicate v 0
      mapM_ (\(k,c) -> do MV.write mv k c) 
            [(j, z * UV.foldl1' (*) (UV.imap (\p i -> lrr V.! p UV.! i) ivv)) | 
              !j <- [0 .. v-1], let !ivv = R.fromIndex svv j]
      return mv
    mempty = Map.empty
    vempty = V.empty
    llrr = R.fromListUnboxed
    vsempty = UV.empty
    vvll = V.toList
 
histogramRepaRedVecsIndependent_u :: HistogramRepaRedVec -> HistogramRepaVec 
histogramRepaRedVecsIndependent_u xxv = HistogramRepaVec vvv mvv z svv vaa
  where
    HistogramRepaRedVec vvv mvv z svv !vxx = xxv
    !v = R.size svv
    !n = R.rank svv
    !p = V.length vxx
    !vaa = runST $ do
      vaa <- V.replicateM p (MV.replicate v 0)
      rvv <- newSTRef (UV.replicate n 0)
      if z /= 1
        then do 
          forM_ [0 .. v-1] $ (\j -> do 
            ivv <- readSTRef rvv
            forM_ [0 .. p-1] $ (\k -> do 
              let !xx = V.unsafeIndex vxx k
              MV.unsafeWrite (V.unsafeIndex vaa k) j 
                (UV.foldl' (*) z (UV.imap (\l i -> UV.unsafeIndex (V.unsafeIndex xx l) i) ivv)))
            writeSTRef rvv (incIndex svv ivv))
        else do 
          forM_ [0 .. v-1] $ (\j -> do 
            ivv <- readSTRef rvv
            forM_ [0 .. p-1] $ (\k -> do 
              let !xx = V.unsafeIndex vxx k
              MV.unsafeWrite (V.unsafeIndex vaa k) j 
                (UV.foldl1' (*) (UV.imap (\l i -> UV.unsafeIndex (V.unsafeIndex xx l) i) ivv)))
            writeSTRef rvv (incIndex svv ivv))
      V.mapM UV.unsafeFreeze vaa

histogramRepaRedVecsIndependent_u_2 :: HistogramRepaRedVec -> HistogramRepaVec 
histogramRepaRedVecsIndependent_u_2 xxv = HistogramRepaVec vvv mvv z svv vaa
  where
    HistogramRepaRedVec vvv mvv z svv !vxx = xxv
    !v = R.size svv
    !n = R.rank svv
    !p = V.length vxx
    !vaa = runST $ do
      vaa <- V.replicateM p (MV.replicate v 0)
      rvv <- newSTRef (UV.replicate n 0)
      forM_ [0 .. v-1] $ (\j -> do 
        ivv <- readSTRef rvv
        forM_ [0 .. p-1] $ (\k -> do 
          let !xx = V.unsafeIndex vxx k
          MV.unsafeWrite (V.unsafeIndex vaa k) j (UV.foldl' (*) z (UV.imap (\l i -> UV.unsafeIndex (V.unsafeIndex xx l) i) ivv)))
        writeSTRef rvv (incIndex svv ivv))
      V.mapM UV.unsafeFreeze vaa

histogramRepaRedVecsIndependent_u_1 :: HistogramRepaRedVec -> HistogramRepaVec 
histogramRepaRedVecsIndependent_u_1 xxv = HistogramRepaVec vvv mvv z svv vaa
  where
    HistogramRepaRedVec vvv mvv z svv !vxx = xxv
    !v = R.size svv
    !n = R.rank svv
    !p = V.length vxx
    !vaa = runST $ do
      vaa <- V.replicateM p (MV.replicate v 0)
      rvv <- newSTRef (UV.replicate n 0)
      forM_ [0 .. v-1] $ (\j -> do 
        ivv <- readSTRef rvv
        forM_ [0 .. p-1] $ (\k -> do 
          let !xx = vxx V.! k
          MV.unsafeWrite (vaa V.! k) j (UV.foldl' (*) z (UV.imap (\l i -> xx V.! l UV.! i) ivv)))
        writeSTRef rvv (incIndex svv ivv))
      V.mapM UV.unsafeFreeze vaa

varsHistogramRepaVecsRollVec_u :: Int -> HistogramRepaVec -> (HistogramRepaVec, (UV.Vector Int, UV.Vector Int))
varsHistogramRepaVecsRollVec_u !u !rrv = (HistogramRepaVec vvv mvv 1 syy vbb, (rs,rt))
  where
    HistogramRepaVec vvv mvv _ svv vaa = rrv
    !n = rank svv
    !p = V.length vaa
    !d = UV.unsafeIndex svv u
    (!rs,!rt) = UV.unzip $ UV.fromList [(s,t) | s <- [1..d-1], t <- [0..d-2], s>t]
    !r = UV.length rs
    !suu = UV.take u svv UV.++ UV.drop (u+1) svv
    !x = R.size suu
    !syy = svv UV.// [(u,r)]
    !w = x*r
    !vbb = runST $ do
      vbb <- V.replicateM p (MV.replicate w 0)
      !iuu <- MV.replicate (n-1) 0
      forM_ [0 .. x-1] $ (\_ -> do 
        forM_ [0 .. r-1] $ (\q -> do 
          !i <- toIndexInsertM u r q suu iuu
          !s <- toIndexInsertM u d (UV.unsafeIndex rs q) suu iuu
          !t <- toIndexInsertM u d (UV.unsafeIndex rt q) suu iuu
          forM_ [0 .. p-1] $ (\l -> do 
            let !aa = V.unsafeIndex vaa l
            MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex aa s + UV.unsafeIndex aa t)))
        incIndexM_ suu iuu)
      V.mapM UV.unsafeFreeze vbb

varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u :: 
  Int -> Int -> Int -> Int -> HistogramRepaVec -> HistogramRepaVec -> HistogramRepaVec
varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u !u !s !t !q !rrv !ssv = 
  HistogramRepaVec vvv mvv z sbb vbb
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    HistogramRepaVec _ _ _ syy vyy = ssv
    !n = rank svv
    !p = V.length vaa
    !d = UV.unsafeIndex svv u
    !r = UV.unsafeIndex syy u
    !c = d-1
    !sbb = svv UV.// [(u,c)]
    !w = R.size sbb
    !suu = UV.take u svv UV.++ UV.drop (u+1) svv
    !x = R.size suu
    !vbb = runST $ do
      vbb <- V.replicateM p (MV.replicate w 0)
      !iuu <- MV.replicate (n-1) 0
      forM_ [0 .. x-1] $ (\_ -> do 
        forM_ [0 .. s-1] $ (\h -> do
          !i <- toIndexInsertM u c h suu iuu
          if h /= t
            then do 
              !f <- toIndexInsertM u d h suu iuu
              forM_ [0 .. p-1] $ (\l -> do 
                MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex (V.unsafeIndex vaa l) f))
            else do
              !f <- toIndexInsertM u r q suu iuu
              forM_ [0 .. p-1] $ (\l -> do 
                MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex (V.unsafeIndex vyy l) f)))
        if s+1 <= d-1
          then
            forM_ [s+1 .. d-1] $ (\h -> do 
              !f <- toIndexInsertM u d h suu iuu
              !i <- toIndexInsertM u c (h-1) suu iuu
              forM_ [0 .. p-1] $ (\l -> do 
                MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex (V.unsafeIndex vaa l) f)))
          else
            return ()
        incIndexM_ suu iuu)
      V.mapM UV.unsafeFreeze vbb

varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u_1 :: 
  Int -> Int -> Int -> Int -> HistogramRepaVec -> HistogramRepaVec -> HistogramRepaVec
varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u_1 !u !s !t !q !rrv !ssv = 
  HistogramRepaVec vvv mvv z sbb vbb
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    HistogramRepaVec _ _ _ syy vyy = ssv
    !n = rank svv
    !p = V.length vaa
    !d = UV.unsafeIndex svv u
    !r = UV.unsafeIndex syy u
    !c = d-1
    !sbb = svv UV.// [(u,c)]
    !w = R.size sbb
    !suu = UV.take u svv UV.++ UV.drop (u+1) svv
    !x = R.size suu
    !vbb = runST $ do
      vbb <- V.replicateM p (MV.replicate w 0)
      !iuu <- MV.replicate (n-1) 0
      forM_ [0 .. x-1] $ (\_ -> do 
        if 0 <= t-1
          then
            forM_ [0 .. t-1] $ (\h -> do 
              !f <- toIndexInsertM u d h suu iuu
              !i <- toIndexInsertM u c h suu iuu
              forM_ [0 .. p-1] $ (\l -> do 
                MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex (V.unsafeIndex vaa l) f)))
          else
            return ()
        do 
          !f <- toIndexInsertM u r q suu iuu
          !i <- toIndexInsertM u c t suu iuu
          forM_ [0 .. p-1] $ (\l -> do 
            MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex (V.unsafeIndex vyy l) f))
        if t+1 <= s-1
          then
            forM_ [t+1 .. s-1] $ (\h -> do 
              !f <- toIndexInsertM u d h suu iuu
              !i <- toIndexInsertM u c h suu iuu
              forM_ [0 .. p-1] $ (\l -> do 
                MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex (V.unsafeIndex vaa l) f)))
          else
            return ()
        if s+1 <= d-1
          then
            forM_ [s+1 .. d-1] $ (\h -> do 
              !f <- toIndexInsertM u d h suu iuu
              !i <- toIndexInsertM u c (h-1) suu iuu
              forM_ [0 .. p-1] $ (\l -> do 
                MV.unsafeWrite (V.unsafeIndex vbb l) i (UV.unsafeIndex (V.unsafeIndex vaa l) f)))
          else
            return ()
        incIndexM_ suu iuu)
      V.mapM UV.unsafeFreeze vbb

foreign import ccall unsafe "arrayHistoryPairsRollMax_u" arrayHistoryPairsRollMax_u :: 
  CLLong -> CLLong -> Ptr CLLong -> CLLong -> CLLong -> 
  Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> 
  Ptr CLLong -> IO (CLLong)

histogramRepaVecsRollMax :: HistogramRepaVec -> (V.Vector (UV.Vector Int),Integer)
histogramRepaVecsRollMax rrv  = (tt, toInteger q)
  where
    HistogramRepaVec _ _ _ svv vaa = rrv
    [aa, aax, bb, bbx] = V.toList vaa 
    !vsaa = SV.unsafeCast (UV.convert aa) :: SV.Vector CDouble
    !vsbb = SV.unsafeCast (UV.convert bb) :: SV.Vector CDouble
    !vsaax = SV.unsafeCast (UV.convert aax) :: SV.Vector CDouble
    !vsbbx = SV.unsafeCast (UV.convert bbx) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !v = R.size svv
    !n = rank svv
    !d = UV.maximum svv
    !nd = n*d
    (!ppm,!q) = unsafePerformIO $ do
      let vsppm = SV.replicate nd 0
      mppm <- SV.unsafeThaw vsppm
      q <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vsaa $ \paa -> do
        SV.unsafeWith vsbb $ \pbb -> do
        SV.unsafeWith vsaax $ \paax -> do
        SV.unsafeWith vsbbx $ \pbbx -> do
        SMV.unsafeWith mppm $ \pmppm -> do
          arrayHistoryPairsRollMax_u (fromIntegral v) (fromIntegral n) psvv (fromIntegral d) (fromIntegral nd) paa paax pbb pbbx pmppm
      vsppm' <- SV.unsafeFreeze mppm
      return ((SV.convert (SV.unsafeCast vsppm' :: SV.Vector Int)),q)
    !tt = V.map (\(i,e) -> UV.take e (UV.drop (d*i) ppm)) (V.indexed (UV.convert svv))

historyRepasTransformRepasApply_u :: HistoryRepa -> TransformRepa -> HistoryRepa 
historyRepasTransformRepasApply_u aa tt = HistoryRepa vbb mbb sbb rbb
  where
    HistoryRepa _ maa saa raa = aa
    TransformRepa vtt mtt w d rtt = tt
    Z :. _ :. (!z) = R.extent raa
    vbb = V.singleton w
    mbb = Map.singleton w 0
    sbb = UV.singleton (fromIntegral d)
    !pkk = V.convert $ V.map (maa Map.!) vtt
    !skk = UV.unsafeBackpermute saa pkk
    !utt = R.toUnboxed rtt 
    !rbb = R.fromUnboxed ((Z :. 1 :. z) :: DIM2) $ UV.create $ do
      ubb <- MV.replicate z 0
      mapM_ (\(k,j) -> do MV.unsafeWrite ubb j (UV.unsafeIndex utt k)) 
            [(R.toIndex skk (UV.map (\i -> fromIntegral (raa R.! (Z :. i :. j))) pkk),j) | !j <- [0 .. z-1]]
      return ubb

historyRepasListTransformRepasApply :: HistoryRepa -> V.Vector TransformRepa -> HistoryRepa 
historyRepasListTransformRepasApply aa ff = 
  historyRepasListTransformRepasApply_u aa $ listVariablesListTransformRepasSort vaa ff
  where
    HistoryRepa vaa _ _ _ = aa

listVariablesListTransformRepasSort :: V.Vector Variable -> V.Vector TransformRepa -> V.Vector TransformRepa 
listVariablesListTransformRepasSort vv ff = next vv' ff' V.empty
  where
    vv' = Set.fromList $ V.toList vv
    ff' = V.map (\tt -> (tt,und tt)) $ V.filter (\tt -> der tt `Set.notMember` vv') ff
    next vv ff gg
      | i' /= Nothing = next (der tt `Set.insert` vv) (rem i ff) (gg `V.snoc` tt)
      | otherwise = gg
      where 
        i' = V.findIndex (\(_,xx) -> xx `Set.isSubsetOf` vv) ff
        i = fromJust i'
        (tt,_) = ff V.! i
    rem i ff = V.take i ff V.++ V.drop (i+1) ff
    und tt =  der tt `Set.delete` vars tt 
    vars = Set.fromList . V.toList . transformRepasVectorVar
    der = transformRepasVarDerived

historyRepasListTransformRepasApply_u :: HistoryRepa -> V.Vector TransformRepa -> HistoryRepa 
historyRepasListTransformRepasApply_u aa ff = HistoryRepa vbb mbb sbb rbb
  where
    HistoryRepa vaa _ saa raa = aa
    Z :. (!n) :. (!z) = R.extent raa
    !m = V.length ff
    !p = n+m
    !qaa = R.fromUnboxed ((Z :. p :. z) :: DIM2) $ R.toUnboxed raa UV.++ UV.replicate (m*z) 0
    !raa' = computeS (R.transpose qaa)
    vbb = vaa V.++ V.map transformRepasVarDerived ff
    mbb = Map.fromList $ zip (V.toList vbb) [0..]
    !sbb = saa UV.++ V.convert (V.map (fromIntegral . transformRepasValency) ff)
    !rbb' = R.fromUnboxed ((Z :. z :. p) :: DIM2) $ UV.create $ do
      qbb <- UV.unsafeThaw (R.toUnboxed raa')
      forM_ [0 .. m-1] $ (\q -> do 
        let TransformRepa vtt _ w _ rtt = V.unsafeIndex ff q
        let !x = mbb Map.! w
        let !utt = R.toUnboxed rtt
        let !pkk = V.convert $ V.map (mbb Map.!) vtt
        let !skk = UV.unsafeBackpermute sbb pkk
        forM_ [0 .. z-1] $ (\j -> do 
          let !r = p*j
          !i <- toIndexPermOffsetM skk pkk r qbb
          MV.unsafeWrite qbb (r+x) (UV.unsafeIndex utt i))) 
      return qbb
    !rbb = computeS (R.transpose rbb')

systemsFudsHistoryRepasMultiply :: System -> Fud -> HistoryRepa -> Maybe HistoryRepa 
systemsFudsHistoryRepasMultiply uu ff aa
  | gg' /= Nothing = Just $ historyRepasListTransformRepasApply aa gg 
  | otherwise = Nothing
  where
    gg' = mapM (systemsTransformsTransformRepa uu) $ Set.toList $ fudsSetTransform ff
    gg = V.fromList $ fromJust gg'

systemsFudsHistoryRepasMultiply_u :: System -> Fud -> HistoryRepa -> HistoryRepa 
systemsFudsHistoryRepasMultiply_u uu ff aa = historyRepasListTransformRepasApply aa gg
  where
    gg = V.fromList $ List.map (systemsTransformsTransformRepa_u uu) $ Set.toList $ fudsSetTransform ff

systemsDecompFudsHistoryRepasMultiply :: System -> DecompFud -> HistoryRepa -> Tree ((State,Fud),HistoryRepa)
systemsDecompFudsHistoryRepasMultiply uu df aa = apply (dfzz df) (vars aa) aa
  where
    apply :: Tree (State,Fud) -> Set.Set Variable -> HistoryRepa -> Tree ((State,Fud),HistoryRepa)
    apply zz vv aa = Tree $ llmm $ [(((ss,ff), bb), apply yy vv bb) | 
      size aa > 0, ((ss,ff),yy) <- zzll zz, let aa' = select uu ss aa, let ww = fder ff, 
      let bb = if size aa' > 0 then (applyFud uu ff aa' `red` (vv `cup` ww)) else empty]
    fder = fudsDerived
    llhh ll = fromJust $ listsHistory ll
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    sshr uu ss = hhhr uu $ llhh [(IdInt 1, ss)]
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (sshr uu ss) hh
    applyFud = systemsFudsHistoryRepasMultiply_u
    red aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    dfzz = decompFudsTreePairStateFud
    size = historyRepasSize
    vars = historyRepasSetVariable
    empty = historyRepaEmpty
    zzll (Tree mm) = mmll mm
    llmm = Map.fromList
    mmll = Map.toList
    cup = Set.union

systemsDecompFudsHistoryRepasMultiply_r :: 
  System -> DecompFud -> HistoryRepa -> Tree (((State,Fud),(HistoryRepa,V.Vector TransformRepa)),HistoryRepa)
systemsDecompFudsHistoryRepasMultiply_r uu df aa = apply (dfzz df) aa
  where
    vv = vars aa
    HistoryRepa vaa _ _ _ = aa
    apply zz aa = Tree $ llmm $ [((((ss,ff),(sr,fr)), bb), apply yy bb) | 
      size aa > 0, ((ss,ff),yy) <- zzll zz, let sr = sshr uu ss, let aa' = select sr aa, 
      let ww = fder ff, let fr = frsort vaa (V.map (tttr uu) (ffvv ff)),
      let bb = if size aa' > 0 then (applyFud aa' fr `red` (vv `cup` ww)) else empty]
    fder = fudsDerived
    llhh ll = fromJust $ listsHistory ll
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    sshr uu ss = hhhr uu $ llhh [(IdInt 1, ss)]
    select sr hh = historyRepasHistoryRepasHistoryRepaSelection_u sr hh
    applyFud = historyRepasListTransformRepasApply_u
    red aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    frsort = listVariablesListTransformRepasSort
    tttr = systemsTransformsTransformRepa_u
    ffvv = V.fromList . Set.toList . fudsSetTransform
    dfzz = decompFudsTreePairStateFud
    size = historyRepasSize
    vars = historyRepasSetVariable
    empty = historyRepaEmpty
    zzll (Tree mm) = mmll mm
    llmm = Map.fromList
    mmll = Map.toList
    cup = Set.union

systemsDecompFudsHistoryRepasSetVariablesListHistogramLeaf :: 
  System -> DecompFud -> HistoryRepa -> Set.Set Variable ->[Histogram]
systemsDecompFudsHistoryRepasSetVariablesListHistogramLeaf uu df hh ll = 
  [trim (aa `mul` unit ss `red` ll) | (((_,ff),hr),yy) <- mult uu df hh, let ww = fder ff,
    let aa = araa uu (hr `hrred` (ww `cup` ll)), 
    let qq = schild yy, (ss,a) <- aall (aa `red` ww), a > 0, ss `Set.notMember` qq]
  where
    schild yy = Set.map (\((ss,_),_) -> ss) $ treesRoots yy
    mult uu df hh = Set.toList $ treesNodes $ systemsDecompFudsHistoryRepasMultiply uu df hh
    hrred hh vv = setVarsHistoryRepasReduce 1 vv hh
    araa uu rr = fromJust $ systemsHistogramRepasHistogram uu rr
    fder = fudsDerived
    aall = histogramsList
    unit = fromJust . setStatesHistogramUnit . Set.singleton
    mul = pairHistogramsMultiply
    trim = histogramsTrim
    red aa vv = setVarsHistogramsReduce vv aa
    cup = Set.union

systemsDecompFudsHistoryRepasHistoriesQuery :: System -> DecompFud -> HistoryRepa -> History -> Tree ((State,Fud),HistoryRepa)
systemsDecompFudsHistoryRepasHistoriesQuery uu df hh qq = query uu (mult uu df hh) (vars hh) qq
  where
    mult = systemsDecompFudsHistoryRepasMultiply
    query = systemsDecompFudMultipliesHistoryRepasHistoriesQuery
    vars = historyRepasSetVariable

systemsDecompFudMultipliesHistoryRepasHistoriesQuery :: System -> Tree ((State,Fud),HistoryRepa) -> Set.Set Variable -> History -> Tree ((State,Fud),HistoryRepa)
systemsDecompFudMultipliesHistoryRepasHistoriesQuery uu zz vv qq = query zz (hhaa qq)
  where
    kk = vars qq
    query :: Tree ((State,Fud),HistoryRepa) -> Histogram -> Tree ((State,Fud),HistoryRepa)
    query zz qq = Tree $ llmm $ [(((ss,ff), hr'), query yy rr) | 
      (((ss,ff),hr),yy) <- zzll zz, let ww = fder ff, let xx = fund ff, 
      let qq' = qq `mul` (single ss 1), size qq' > 0,
      let rr = if xx `subset` kk then applyFud (vv `cup` ww) ff qq' else applyHis (vv `cup` kk) (vv `cup` ww) (ffqq ff) qq',
      let hr' = select (hhhr uu (aahh (rr `red` ww))) hr]
    fder = fudsDerived
    fund = fudsUnderlying      
    ffqq = fudsSetHistogram
    single ss c = fromJust $ histogramSingleton ss c
    mul = pairHistogramsMultiply
    applyFud ww ff aa = fromJust $ setVarsFudHistogramsApply ww ff aa
    applyHis = setVarsSetVarsSetHistogramsHistogramsApply
    llhh ll = fromJust $ listsHistory ll
    aahh aa = llhh [(IdInt i, ss) | (ss,i) <- zip [ss | (ss,a) <- aall aa, i <- [1..truncate a]] [1..]]
    hhaa hh = historiesHistogram hh
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    select = historyRepasHistoryRepasHistoryRepaSelection_u
    red = flip setVarsHistogramsReduce
    vars = historiesVars
    size = histogramsSize
    aall = histogramsList
    zzll (Tree mm) = mmll mm
    mmll = Map.toList
    llmm = Map.fromList
    cup = Set.union
    subset = Set.isSubsetOf

systemsDecompFudsHistoryRepasHistoryRepasQuery :: 
  System -> DecompFud -> HistoryRepa -> HistoryRepa -> Tree ((State,Fud),HistoryRepa)
systemsDecompFudsHistoryRepasHistoryRepasQuery uu df hh qq = query uu (mult uu df hh) qq
  where
    mult = systemsDecompFudsHistoryRepasMultiply_r
    query = systemsDecompFudMultipliesHistoryRepasHistoryRepasQuery
 
systemsDecompFudMultipliesHistoryRepasHistoryRepasQuery :: 
  System -> Tree (((State,Fud),(HistoryRepa,V.Vector TransformRepa)),HistoryRepa) -> HistoryRepa -> 
  Tree ((State,Fud),HistoryRepa)
systemsDecompFudMultipliesHistoryRepasHistoryRepasQuery uu zz qq = query zz qq
  where
    kk = vars qq
    query zz qq = Tree $ llmm $ [(((ss,ff), hr'), query yy rr) | 
      ((((ss,ff),(sr,fr)),hr),yy) <- zzll zz,   
      let qq' = select sr qq, size qq' > 0, fund ff `subset` kk, let ww = fder ff, 
      let rr = applyFud qq' fr `red` (kk `cup` ww), let hr' = select (rr `red` ww) hr]
    fder = fudsDerived
    fund = fudsUnderlying      
    applyFud = historyRepasListTransformRepasApply_u
    select = historyRepasHistoryRepasHistoryRepaSelection_u
    red hh vv = setVarsHistoryRepasHistoryRepaReduced vv hh
    size = historyRepasSize
    vars = historyRepasSetVariable
    aall = histogramsList
    zzll (Tree mm) = mmll mm
    mmll = Map.toList
    llmm = Map.fromList
    cup = Set.union
    subset = Set.isSubsetOf

systemsDecompFudMultipliesHistoryRepasHistoryRepasQueryAny :: 
  System -> Tree (((State,Fud),(HistoryRepa,V.Vector TransformRepa)),HistoryRepa) -> HistoryRepa -> 
  [((State,Fud),HistoryRepa)]
systemsDecompFudMultipliesHistoryRepasHistoryRepasQueryAny uu zz qq = query zz qq
  where
    kk = vars qq
    query zz qq 
      | pp /= [] = let (xx,(yy,rr)) = head pp in xx : query yy rr
      | otherwise = []
      where 
        pp = [(((ss,ff), hr'),(yy,rr)) | ((((ss,ff),(sr,fr)),hr),yy) <- zzll zz,   
          let qq' = select sr qq, size qq' > 0, fund ff `subset` kk, let ww = fder ff, 
          let rr = applyFud qq' fr `red` (kk `cup` ww), let hr' = select (rr `red` ww) hr]
    fder = fudsDerived
    fund = fudsUnderlying      
    applyFud = historyRepasListTransformRepasApply_u
    select = historyRepasHistoryRepasHistoryRepaSelection_u
    red hh vv = setVarsHistoryRepasHistoryRepaReduced vv hh
    size = historyRepasSize
    vars = historyRepasSetVariable
    aall = histogramsList
    zzll (Tree mm) = mmll mm
    mmll = Map.toList
    cup = Set.union
    subset = Set.isSubsetOf

systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest :: 
  System -> DecompFud -> HistoryRepa -> HistoryRepa -> Set.Set Variable -> [(State,(Histogram,(Bool,Bool)))]
systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest uu df hha hht kk = 
  [(l, (a,(b,c))) | i <- [0 .. zt-1], let !hr = hrev [i] hht, let !l = label hr, let (!a,(!b,!c)) = result hr]
  where
    vv = vars hha
    ll = vv `minus` kk 
    zza = systemsDecompFudsHistoryRepasMultiply_r uu df hha
    zt = hrsize hht
    rra = araa $ hha `hrred` ll
    label hr
      | xx /= []  = head xx
      | otherwise = sempty
      where
        xx = fst $ unzip $ aall $ araa $ hr `hrred` ll
    result hr
      | pp' == [] = (rra,(False,False))
      | otherwise = (araa (head pp' `hrred` ll), (length pp' == length pp,True))
      where
        pp = snd $ unzip $ query uu zza hr
        pp' = dropWhile (\hr -> hrsize hr == 0) $ reverse pp
    query = systemsDecompFudMultipliesHistoryRepasHistoryRepasQueryAny
    hrred hh vv = setVarsHistoryRepasReduce 1 vv hh
    hrsize = historyRepasSize
    hrev = eventsHistoryRepasHistoryRepaSelection
    araa rr = trim $ fromJust $ systemsHistogramRepasHistogram uu rr
    vars = historyRepasSetVariable
    trim = histogramsTrim
    aall = histogramsList
    sempty = stateEmpty
    minus = Set.difference

systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_1 :: 
  System -> DecompFud -> HistoryRepa -> HistoryRepa -> Set.Set Variable -> [(State,(Histogram,Bool))]
systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_1 uu df hha hht kk = 
  [(label hr, result hr) | i <- [0 .. zt-1], let hr = hrev [i] hht]
  where
    vv = vars hha
    ll = vv `minus` kk 
    zza = systemsDecompFudsHistoryRepasMultiply uu df hha
    zt = historyRepasSize hht
    rra = araa $ hha `hrred` ll
    label hr
      | xx /= []  = head xx
      | otherwise = sempty
      where
        xx = fst $ unzip $ aall $ araa $ hr `hrred` ll
    result hr
      | Set.null qq = (rra,False)
      | otherwise = (last pp', length pp' == length pp)
      where
        qq = treesPaths $ query uu zza vv (hrhh uu hr)
        pp = rra : List.map (\(_,hr) -> araa (hr `hrred` ll)) (Set.findMin qq)
        pp' = filter (\aa -> size aa > 0) pp
    query = systemsDecompFudMultipliesHistoryRepasHistoriesQuery
    hrhh uu hr = fromJust $ systemsHistoryRepasHistory_u uu hr
    hrred hh vv = setVarsHistoryRepasReduce 1 vv hh
    hrev = eventsHistoryRepasHistoryRepaSelection
    araa rr = trim $ fromJust $ systemsHistogramRepasHistogram uu rr
    vars = historyRepasSetVariable
    trim = histogramsTrim
    size = histogramsSize
    aall = histogramsList
    sempty = stateEmpty
    minus = Set.difference

systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_2 :: 
  System -> DecompFud -> HistoryRepa -> HistoryRepa -> Set.Set Variable -> [(State,(Histogram,Bool))]
systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_2 uu df hha hht kk = 
  [(l, (a,b)) | i <- [0 .. zt-1], let !hr = hrev [i] hht, let !l = label hr, let (!a,!b) = result hr]
  where
    vv = vars hha
    ll = vv `minus` kk 
    zza = systemsDecompFudsHistoryRepasMultiply uu df hha
    zt = hrsize hht
    rra = araa $ hha `hrred` ll
    label hr
      | xx /= []  = head xx
      | otherwise = sempty
      where
        xx = fst $ unzip $ aall $ araa $ hr `hrred` ll
    result hr
      | Set.null qq = (rra,False)
      | pp' == [] = (rra,False)
      | otherwise = (araa (last pp' `hrred` ll), length pp' == length pp)
      where
        qq = treesPaths $ query uu zza vv (hrhh uu hr)
        pp = snd $ unzip $ Set.findMin qq
        pp' = filter (\hr -> hrsize hr > 0) pp
    query = systemsDecompFudMultipliesHistoryRepasHistoriesQuery
    hrhh uu hr = fromJust $ systemsHistoryRepasHistory_u uu hr
    hrred hh vv = setVarsHistoryRepasReduce 1 vv hh
    hrsize = historyRepasSize
    hrev = eventsHistoryRepasHistoryRepaSelection
    araa rr = trim $ fromJust $ systemsHistogramRepasHistogram uu rr
    vars = historyRepasSetVariable
    trim = histogramsTrim
    aall = histogramsList
    sempty = stateEmpty
    minus = Set.difference

systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_3 :: 
  System -> DecompFud -> HistoryRepa -> HistoryRepa -> Set.Set Variable -> [(State,(Histogram,Bool))]
systemsDecompFudsHistoryRepasHistoryRepasSetVariablesTest_3 uu df hha hht kk = 
  [(l, (a,b)) | i <- [0 .. zt-1], let !hr = hrev [i] hht, let !l = label hr, let (!a,!b) = result hr]
  where
    vv = vars hha
    ll = vv `minus` kk 
    zza = systemsDecompFudsHistoryRepasMultiply_r uu df hha
    zt = hrsize hht
    rra = araa $ hha `hrred` ll
    label hr
      | xx /= []  = head xx
      | otherwise = sempty
      where
        xx = fst $ unzip $ aall $ araa $ hr `hrred` ll
    result hr
      | Set.null qq = (rra,False)
      | pp' == [] = (rra,False)
      | otherwise = (araa (last pp' `hrred` ll), length pp' == length pp)
      where
        qq = treesPaths $ query uu zza hr
        pp = snd $ unzip $ Set.findMin qq
        pp' = filter (\hr -> hrsize hr > 0) pp
    query = systemsDecompFudMultipliesHistoryRepasHistoryRepasQuery
    hrred hh vv = setVarsHistoryRepasReduce 1 vv hh
    hrsize = historyRepasSize
    hrev = eventsHistoryRepasHistoryRepaSelection
    araa rr = trim $ fromJust $ systemsHistogramRepasHistogram uu rr
    vars = historyRepasSetVariable
    trim = histogramsTrim
    aall = histogramsList
    sempty = stateEmpty
    minus = Set.difference

systemsDecompFudsHistoryRepasDecompFudReduced :: System -> DecompFud -> HistoryRepa -> DecompFud
systemsDecompFudsHistoryRepasDecompFudReduced uu df aa = df'
  where
    [(((ss,ff),_),yy)] = zzll $ funcsTreesMap (\(xx,hr) -> (xx,hrsize hr)) $ hrmult uu df aa
    w = least (fder ff)
    df' = zzdf $ llzz [((ss, ff `fdep` w), apply w yy)]
    apply :: Variable -> Tree ((State,Fud),Int) -> Tree (State,Fud)
    apply w zz = Tree $ llmm $ List.map (\(x,(_,yy)) -> (x,yy)) $ mmll $ llmmw larger $ 
      [((ss `red` w, ff `fdep` u),(a,apply u yy)) | (((ss,ff),a),yy) <- zzll zz, let u = least (fder ff)]
    hrmult = systemsDecompFudsHistoryRepasMultiply
    hrsize = historyRepasSize
    fder = fudsDerived
    fdep ff x = fudsVarsDepends ff (sgl x)
    red ss v = setVarsStatesStateFiltered (sgl v) ss
    larger (a,xx1) (b,xx2) = if a>b then (a,xx1) else (b,xx2)
    dfzz = decompFudsTreePairStateFud
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    zzll (Tree mm) = mmll mm
    llzz ll = Tree (llmm ll)
    llmmw = Map.fromListWith
    llmm = Map.fromList
    mmll = Map.toList
    least :: Set.Set a -> a
    least = Set.findMin
    sgl :: a -> Set.Set a
    sgl = Set.singleton

vectorHistoryRepasConcat_u :: V.Vector HistoryRepa -> HistoryRepa 
vectorHistoryRepasConcat_u ll = HistoryRepa vaa maa saa rbb
  where
    HistoryRepa vaa maa saa _ = V.head ll
    rbb = computeS $ V.foldl1 R.append $ V.map (R.delay . historyRepasArray) ll

vectorPairsTop :: Ord a => Int -> V.Vector (a,b) -> V.Vector (a,b)
vectorPairsTop n vv 
  | n <= 0 = V.empty
  | n >= l = vv
  | n == 1 = vv5
  | otherwise = vv4
  where
    l = V.length vv
    vv1 = V.imap (\i (a,b) -> (a,i)) vv     
    vv2 = V.create $ do
      mv <- V.unsafeThaw vv1
      VA.sort mv
      return mv
    vv3 = V.drop (l-n) vv2
    vv4 = V.map (\(_,i) -> V.unsafeIndex vv i) vv3  
    (_,m) = V.maximum vv1
    vv5 = V.singleton (V.unsafeIndex vv m)

setVarsHistoryRepasHistoryRepaReduced :: Set.Set Variable -> HistoryRepa -> HistoryRepa 
setVarsHistoryRepasHistoryRepaReduced kk hh = HistoryRepa vkk mkk skk rr'
  where
    HistoryRepa vvv mvv svv !rr = hh
    vv = llqq $ vvll vvv
    Z :. _ :. z = extent rr
    !vkk = llvv $ qqll (kk `cap` vv)
    mkk = llmm (zip (vvll vkk) [0..])
    !m = V.length vkk
    !pkk = llvu $ vvll $ V.map (mvv Map.!) vkk
    !skk = perm svv pkk
    !vrr = R.toUnboxed rr
    !rr' = R.fromUnboxed ((Z :. m :. z) :: DIM2) $ UV.concat $ List.map (\p -> UV.slice (p*z) z vrr) $ vull pkk
    perm = UV.unsafeBackpermute
    llmm = Map.fromList
    qqll = Set.toList
    llqq = Set.fromList
    cap = Set.intersection
    llvv = V.fromList
    vvll = V.toList
    llvu = UV.fromList
    vull = UV.toList

eventsHistoryRepasHistoryRepaSelection :: [Int] -> HistoryRepa -> HistoryRepa 
eventsHistoryRepasHistoryRepaSelection ll hh = HistoryRepa vvv mvv svv rr'
  where
    HistoryRepa vvv mvv svv !rr = hh
    Z :. n :. z = R.extent rr
    ll' = List.filter (\i -> i >= 0 && i < z) ll
    !y = List.length ll'
    !vrr = R.toUnboxed $ R.computeS $ R.transpose rr
    !rr' = R.computeS $ R.transpose $ R.fromUnboxed ((Z :. y :. n) :: DIM2) $ 
      UV.concat $ List.map (\i -> UV.slice (i*n) n vrr) $ ll'

historyRepasHistoryRepasHistoryRepaSelection_u :: HistoryRepa -> HistoryRepa -> HistoryRepa 
historyRepasHistoryRepasHistoryRepaSelection_u ss hh 
  | ss == empty || hh == empty = empty
  | V.null vss = hh 
  | otherwise = HistoryRepa vhh mhh shh rhh'
  where
    HistoryRepa vss _ _ !rss = ss
    HistoryRepa vhh mhh shh !rhh = hh
    Z :. m :. y = R.extent rss
    Z :. n :. z = R.extent rhh
    !pss = llvu $ vvll $ V.map (mhh Map.!) vss
    !xss = R.toUnboxed $ R.computeS $ R.transpose rss
    !xhh = R.toUnboxed $ R.computeS $ R.transpose rhh
    !lss = List.map (\i -> UV.slice (i*m) m xss) [0..y-1]
    !lhh = List.filter (\ii -> let pp = perm ii pss in any (==pp) lss) $ List.map (\i -> UV.slice (i*n) n xhh) [0..z-1]
    !z' = List.length lhh
    !rhh' = R.computeS $ R.transpose $ R.fromUnboxed ((Z :. z' :. n) :: DIM2) $ UV.concat lhh
    empty = historyRepaEmpty
    perm = UV.unsafeBackpermute
    vvll = V.toList
    llvu = UV.fromList

historyRepasSize :: HistoryRepa -> Int 
historyRepasSize hh = z
  where
    HistoryRepa _ _ _ rhh = hh
    Z :. _ :. z = R.extent rhh

historyRepasSetVariable :: HistoryRepa -> Set.Set Variable 
historyRepasSetVariable (HistoryRepa vaa _ _ _) = llqq $ vvll vaa
  where
    llqq = Set.fromList
    vvll = V.toList

historyRepasDimension :: HistoryRepa -> Int 
historyRepasDimension hh = n
  where
    HistoryRepa _ _ _ rhh = hh
    Z :. n :. _ = R.extent rhh

historyRepasListsList :: HistoryRepa -> [[Int16]] 
historyRepasListsList hh = [UV.toList (UV.slice (i*n) n vrr) | i <- [0..z-1]]
  where
    HistoryRepa _ _ _ !rr = hh
    Z :. n :. z = R.extent rr
    !vrr = R.toUnboxed $ R.computeS $ R.transpose rr

systemsListVariablesListsListsHistoryRepa_u :: System -> [Variable] -> [[Int16]] -> HistoryRepa
systemsListVariablesListsListsHistoryRepa_u uu vv xx = HistoryRepa (llvv vv) mvv (llvu sh) xx'
  where
    mvv = llmm (zip vv [0..])
    sh = [uval uu v | v <- vv]
    z = length xx
    n = length vv
    xx' = R.computeS $ R.transpose $ R.fromUnboxed ((Z :. z :. n) :: DIM2) $ UV.concat $ List.map (UV.fromListN n) xx
    uval uu v = Set.size $ fromJust $ systemsVarsSetValue uu v
    llmm = Map.fromList
    llvv = V.fromList
    llvu = UV.fromList

systemsListVariablesListsListsHistoryRepa :: System -> [Variable] -> [[Int16]] -> Maybe HistoryRepa
systemsListVariablesListsListsHistoryRepa uu vv xx 
  | not (llqq vv `subset` uvars uu) = Nothing
  | not $ all (\ll -> length ll == n) xx = Nothing
  | otherwise = Just $ systemsListVariablesListsListsHistoryRepa_u uu vv xx
  where
    n = length vv
    uvars = systemsVars
    subset = Set.isSubsetOf
    llqq = Set.fromList

setSetVarsHistogramRepaVecsPartitionIndependentVec_u :: 
  Set.Set (Set.Set Variable) -> HistogramRepaVec -> HistogramRepaVec 
setSetVarsHistogramRepaVecsPartitionIndependentVec_u pp rrv = 
  HistogramRepaVec vyy myy z syy vbb
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !p = V.length vaa
    !f = 1 / z
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !m = V.length vpp
    yy = List.map VarIndex [0 .. m-1]
    vyy = llvv yy
    myy = llmm (zip yy [0..])
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !spp = V.map (\pcc -> perm svv pcc) ppp 
    !syy = V.convert $ V.map R.size spp
    !vxx = runST $ do
      vxx <- V.replicateM p (V.generateM m (\i -> MV.replicate (UV.unsafeIndex syy i) 0))
      !ivv <- MV.replicate n 0
      forM_ [0 .. v-1] $ (\j -> do 
        forM_ [0 .. m-1] $ (\k -> do 
          !i <- toIndexPermM (V.unsafeIndex spp k) (V.unsafeIndex ppp k) ivv
          forM_ [0 .. p-1] $ (\l -> do 
            let !mv = V.unsafeIndex (V.unsafeIndex vxx l) k
            c <- MV.unsafeRead mv i
            let !a = UV.unsafeIndex (V.unsafeIndex vaa l) j
            MV.unsafeWrite mv i (c+a)))
        incIndexM_ svv ivv)
      if f /= 1 
        then do
          forM_ [0 .. p-1] $ (\l -> do 
            forM_ [0 .. m-1] $ (\k -> do 
              let !mv = V.unsafeIndex (V.unsafeIndex vxx l) k
              forM_ [0..(MV.length mv)-1] $ (\i -> do 
                c <- MV.unsafeRead mv i
                MV.unsafeWrite mv i (c*f))))
        else do
          return ()
      V.mapM (V.mapM UV.unsafeFreeze) vxx
    !vbb = runST $ do
      vbb <- V.replicateM p (MV.replicate v 0)
      ryy <- newSTRef (UV.replicate m 0)
      if z /= 1
        then do 
          forM_ [0 .. v-1] $ (\j -> do 
            iyy <- readSTRef ryy
            forM_ [0 .. p-1] $ (\k -> do 
              let !xx = V.unsafeIndex vxx k
              MV.unsafeWrite (V.unsafeIndex vbb k) j 
                (UV.foldl' (*) z (UV.imap (\l i -> UV.unsafeIndex (V.unsafeIndex xx l) i) iyy)))
            writeSTRef ryy (incIndex syy iyy))
        else do 
          forM_ [0 .. v-1] $ (\j -> do 
            iyy <- readSTRef ryy
            forM_ [0 .. p-1] $ (\k -> do 
              let !xx = V.unsafeIndex vxx k
              MV.unsafeWrite (V.unsafeIndex vbb k) j 
                (UV.foldl1' (*) (UV.imap (\l i -> UV.unsafeIndex (V.unsafeIndex xx l) i) iyy)))
            writeSTRef ryy (incIndex syy iyy))
      V.mapM UV.unsafeFreeze vbb
    llmm = Map.fromList
    perm = UV.unsafeBackpermute
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList

foreign import ccall unsafe "listListVarsArrayHistoryPairsPartitionIndependent_u" listListVarsArrayHistoryPairsPartitionIndependent_u :: 
  CDouble -> CLLong -> CLLong -> Ptr CLLong -> CLLong -> CLLong -> 
  Ptr CLLong -> Ptr CLLong -> Ptr CLLong -> Ptr CDouble -> Ptr CDouble -> 
  Ptr CDouble -> Ptr CDouble -> IO ()

setSetVarsHistogramRepaPairStorablesPartitionIndependentPair_u :: 
  Set.Set (Set.Set Variable) -> HistogramRepaVec -> SV.Vector CDouble -> SV.Vector CDouble -> HistogramRepaVec 
setSetVarsHistogramRepaPairStorablesPartitionIndependentPair_u pp rrv !vsaa vsaarr = 
  HistogramRepaVec vyy myy z syy vbb
  where
    HistogramRepaVec vvv mvv z svv vaa = rrv
    !v = R.size svv
    !n = rank svv
    !vpp = llvv $ [qqvv cc | cc <- qqll pp] 
    !m = V.length vpp
    yy = List.map VarIndex [0 .. m-1]
    !vyy = llvv yy
    myy = llmm (zip yy [0..])
    !ppp = V.map (\vcc -> V.convert $ V.map (mvv Map.!) vcc) vpp 
    !syy = V.convert $ V.map R.size $ V.map (\pcc -> perm svv pcc) ppp
    !r = UV.sum syy
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vsppp = SV.unsafeCast (UV.convert (UV.concat (V.toList ppp))) :: SV.Vector CLLong
    !vslyy = SV.unsafeCast (UV.convert (V.map UV.length ppp)) :: SV.Vector CLLong
    !vssyy = SV.unsafeCast (UV.convert syy) :: SV.Vector CLLong
    (!vsbb,!vsbbrr) = unsafePerformIO $ do
      let vsbb = SV.replicate v 0
      let vsbbrr = SV.replicate v 0
      mbb <- SV.unsafeThaw vsbb
      mbbrr <- SV.unsafeThaw vsbbrr
      SV.unsafeWith vssvv $ \psvv -> do
      SV.unsafeWith vslyy $ \plyy -> do
      SV.unsafeWith vssyy $ \psyy -> do
      SV.unsafeWith vsppp $ \pppp -> do
      SV.unsafeWith vsaa $ \paa -> do
      SV.unsafeWith vsaarr $ \paarr -> do
      SMV.unsafeWith mbb $ \pmbb -> do
      SMV.unsafeWith mbbrr $ \pmbbrr -> do
        listListVarsArrayHistoryPairsPartitionIndependent_u (realToFrac z) (fromIntegral v) (fromIntegral n) psvv (fromIntegral m) (fromIntegral r) plyy psyy pppp paa paarr pmbb pmbbrr
      vsbb' <- SV.unsafeFreeze mbb 
      vsbbrr' <- SV.unsafeFreeze mbbrr
      return (vsbb',vsbbrr')
    !vbb = V.map (SV.convert . SV.unsafeCast) $ V.fromList [vsbb,vsbbrr] 
    llmm = Map.fromList
    perm = UV.unsafeBackpermute
    qqvv = llvv . qqll
    llvv = V.fromList
    vvll = V.toList
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList


foreign import ccall unsafe "listListVarsArrayHistoryPairsSetTuplePartitionTop_u" listListVarsArrayHistoryPairsSetTuplePartitionTop_u :: 
  CLLong -> CDouble -> CLLong -> CLLong -> Ptr CLLong -> CLLong -> CDouble -> 
  Ptr CLLong -> Ptr CLLong -> Ptr CLLong -> Ptr CLLong -> Ptr CDouble -> Ptr CDouble -> 
  Ptr CLLong -> IO (CLLong)

parametersHistogramRepaVecsSetTuplePartitionTop_u :: 
  Integer -> Integer -> Integer -> HistogramRepaVec -> Double -> (Set.Set (Set.Set (Set.Set Variable)),Integer)
parametersHistogramRepaVecsSetTuplePartitionTop_u mmax umax pmax rrv y1 = (tt, toInteger q)
  where
    HistogramRepaVec vvv _ z svv vaa = rrv
    [aa, _, aarr, _] = V.toList vaa 
    !vsaa = SV.unsafeCast (UV.convert aa) :: SV.Vector CDouble
    !vsaarr = SV.unsafeCast (UV.convert aarr) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !v = R.size svv
    !n = rank svv
    qq = [(length pp, List.map length pp, rr, concat pp, 
             Set.fromList (List.map (\cc -> Set.fromList (List.map (\p -> vvv V.! p) cc)) pp)) | 
           ll <- tail (foldl (\mm i -> [j:xx | xx <- mm , j <- [0..i-1], j < (fromIntegral mmax), j <= maximum xx + 1]) [[0]] [2..n]), 
           let pp = List.map Set.toList (Map.elems (Map.fromListWith Set.union [(c, Set.singleton p) | (c,p) <- zip ll [0..]])),
           let rr = List.map (\cc -> R.size (perm svv (UV.fromList cc))) pp, and [u <= (fromIntegral umax) | u <- rr]]
    !q = length qq
    !vsqm = SV.unsafeCast (UV.convert (UV.fromList (List.map (\(m,_,_,_,_) -> m) qq))) :: SV.Vector CLLong
    !vsql = SV.unsafeCast (UV.convert (UV.concat (List.map (\(_,ll,_,_,_) -> nlluv n ll) qq))) :: SV.Vector CLLong
    !vsqs = SV.unsafeCast (UV.convert (UV.concat (List.map (\(_,_,ss,_,_) -> nlluv n ss) qq))) :: SV.Vector CLLong
    !vsqp = SV.unsafeCast (UV.convert (UV.concat (List.map (\(_,_,_,pp,_) -> nlluv n pp) qq))) :: SV.Vector CLLong
    (!vstt,!t) = unsafePerformIO $ do
      let vstt = SV.replicate (fromIntegral pmax) 0
      mtt <- SV.unsafeThaw vstt
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vsqm $ \pqm -> do
        SV.unsafeWith vsql $ \pql -> do
        SV.unsafeWith vsqs $ \pqs -> do
        SV.unsafeWith vsqp $ \pqp -> do
        SV.unsafeWith vsaa $ \paa -> do
        SV.unsafeWith vsaarr $ \paarr -> do
        SMV.unsafeWith mtt $ \pmtt -> do
          listListVarsArrayHistoryPairsSetTuplePartitionTop_u (fromIntegral pmax) (realToFrac z) (fromIntegral v) (fromIntegral n) 
            psvv (fromIntegral q) (realToFrac y1) pqm pql pqs pqp paa paarr pmtt
      vstt' <- SV.unsafeFreeze mtt 
      return (SV.take (fromIntegral t) vstt',t)
    !tt = Set.fromList (List.map (\p -> let (_,_,_,_,yy) = qq !! (fromIntegral p) in yy) (SV.toList vstt))
    nlluv n ll = let d = n - length ll in if d > 0 then UV.fromList ll UV.++ UV.replicate d 0 else UV.fromListN n ll
    perm = UV.unsafeBackpermute

parametersHistogramRepaVecsSetTuplePartitionTopByM_u :: 
  Integer -> Integer -> Integer -> HistogramRepaVec -> Double -> (Set.Set (Set.Set (Set.Set Variable)),Integer)
parametersHistogramRepaVecsSetTuplePartitionTopByM_u mmax umax pmax rrv y1 = (tt, toInteger q)
  where
    HistogramRepaVec vvv _ z svv vaa = rrv
    [aa, _, aarr, _] = V.toList vaa 
    !vsaa = SV.unsafeCast (UV.convert aa) :: SV.Vector CDouble
    !vsaarr = SV.unsafeCast (UV.convert aarr) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !v = R.size svv
    !n = rank svv
    qq = [(length pp, List.map length pp, rr, concat pp, 
             Set.fromList (List.map (\cc -> Set.fromList (List.map (\p -> vvv V.! p) cc)) pp)) | 
           ll <- tail (foldl (\mm i -> [j:xx | xx <- mm , j <- [0..i-1], j < (fromIntegral mmax), j <= maximum xx + 1]) [[0]] [2..n]), 
           let pp = List.map Set.toList (Map.elems (Map.fromListWith Set.union [(c, Set.singleton p) | (c,p) <- zip ll [0..]])),
           let rr = List.map (\cc -> R.size (perm svv (UV.fromList cc))) pp, and [u <= (fromIntegral umax) | u <- rr]]
    !q = length qq
    !tt = foldl Set.union Set.empty [parter qq' | m <- [2 .. fromIntegral mmax], let qq' = filter (\(m',_,_,_,_) -> m' == m) qq]
    parter !qq = Set.fromList (List.map (\p -> let (_,_,_,_,yy) = qq !! (fromIntegral p) in yy) (SV.toList vstt))
      where
        !q = length qq
        !vsqm = SV.unsafeCast (UV.convert (UV.fromList (List.map (\(m,_,_,_,_) -> m) qq))) :: SV.Vector CLLong
        !vsql = SV.unsafeCast (UV.convert (UV.concat (List.map (\(_,ll,_,_,_) -> nlluv n ll) qq))) :: SV.Vector CLLong
        !vsqs = SV.unsafeCast (UV.convert (UV.concat (List.map (\(_,_,ss,_,_) -> nlluv n ss) qq))) :: SV.Vector CLLong
        !vsqp = SV.unsafeCast (UV.convert (UV.concat (List.map (\(_,_,_,pp,_) -> nlluv n pp) qq))) :: SV.Vector CLLong
        (!vstt,!t) = unsafePerformIO $ do
          let vstt = SV.replicate (fromIntegral pmax) 0
          mtt <- SV.unsafeThaw vstt
          t <- SV.unsafeWith vssvv $ \psvv -> do
            SV.unsafeWith vsqm $ \pqm -> do
            SV.unsafeWith vsql $ \pql -> do
            SV.unsafeWith vsqs $ \pqs -> do
            SV.unsafeWith vsqp $ \pqp -> do
            SV.unsafeWith vsaa $ \paa -> do
            SV.unsafeWith vsaarr $ \paarr -> do
            SMV.unsafeWith mtt $ \pmtt -> do
              listListVarsArrayHistoryPairsSetTuplePartitionTop_u (fromIntegral pmax) (realToFrac z) (fromIntegral v) (fromIntegral n) 
                psvv (fromIntegral q) (realToFrac y1) pqm pql pqs pqp paa paarr pmtt
          vstt' <- SV.unsafeFreeze mtt 
          return (SV.take (fromIntegral t) vstt',t)
        nlluv n ll = let d = n - length ll in if d > 0 then UV.fromList ll UV.++ UV.replicate d 0 else UV.fromListN n ll
    perm = UV.unsafeBackpermute

parametersSetVarsHistoryRepasSetSetVarsAlignedTop :: Integer -> Integer -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Maybe (V.Vector ((Double,Double,Integer),Set.Set Variable),Integer)
parametersSetVarsHistoryRepasSetSetVarsAlignedTop xmax omax vv hh hhx hhrr hhrrx
  | xmax <= 0 || omax <= 0 || z <= 0 || zrr <= 0 = Nothing
  | not (vhhx == vhh && vhhx == vhh && vhhrr == vhh && vhhrrx == vhh) = Nothing
  | not (vv `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | otherwise = Just $ parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u xmax omax vv hh hhx hhrr hhrrx
  where
    HistoryRepa vhh mvv _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    R.Z R.:. _ R.:. z = R.extent aa
    R.Z R.:. _ R.:. zrr = R.extent aarr

foreign import ccall unsafe "listVarsArrayHistoriesAlignedTop_u" listVarsArrayHistoriesAlignedTop_u
  :: CLLong -> CLLong -> CLLong -> Ptr CLLong -> CLLong -> 
    CLLong -> CLLong -> Ptr CLLong -> Ptr CShort -> Ptr CDouble -> Ptr CShort -> Ptr CDouble -> 
    Ptr CLLong -> Ptr CLLong -> Ptr CDouble -> Ptr CDouble -> Ptr CLLong -> Ptr CLLong -> IO (CLLong)

parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u :: Integer -> Integer -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> (V.Vector ((Double,Double,Integer),Set.Set Variable),Integer) 
parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u xmax omax ww hh hhx hhrr hhrrx = (qq,s)
  where
    HistoryRepa vhh mvv svv aa = hh
    HistogramRepaRed _ _ _ laax = hhx
    HistoryRepa _ _ _ aarr = hhrr
    HistogramRepaRed _ _ _ laarrx = hhrrx
    R.Z R.:. (!n) R.:. (!z) = R.extent aa
    R.Z R.:. _ R.:. (!zrr) = R.extent aarr
    !vww = llvv $ qqll ww 
    !m = V.length vww
    !pww = V.map (mvv Map.!) vww 
    !vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CShort
    !vshhx = SV.unsafeCast (UV.convert (UV.concat (V.toList laax))) :: SV.Vector CDouble
    !vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CShort
    !vshhrrx = SV.unsafeCast (UV.convert (UV.concat (V.toList laarrx))) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vspww = SV.unsafeCast (UV.convert pww) :: SV.Vector CLLong
    (!vsqww1,!vsqww2,!vsts1,!vsts2,!vsts3,!s) = unsafePerformIO $ do
      let vsqww1 = SV.replicate (fromIntegral omax) 0
      let vsqww2 = SV.replicate (fromIntegral omax) 0
      let vsts1 = SV.replicate (fromIntegral omax) 0
      let vsts2 = SV.replicate (fromIntegral omax) 0
      let vsts3 = SV.replicate (fromIntegral omax) 0
      let vsss = SV.replicate 1 0
      mqww1 <- SV.unsafeThaw vsqww1
      mqww2 <- SV.unsafeThaw vsqww2
      mts1 <- SV.unsafeThaw vsts1
      mts2 <- SV.unsafeThaw vsts2
      mts3 <- SV.unsafeThaw vsts3
      mss <- SV.unsafeThaw vsss
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vspww $ \ppww -> do
        SV.unsafeWith vshh $ \phh -> do
        SV.unsafeWith vshhx $ \phhx -> do
        SV.unsafeWith vshhrr $ \phhrr -> do
        SV.unsafeWith vshhrrx $ \phhrrx -> do
        SMV.unsafeWith mqww1 $ \pmqww1 -> do
        SMV.unsafeWith mqww2 $ \pmqww2 -> do
        SMV.unsafeWith mts1 $ \pmts1 -> do
        SMV.unsafeWith mts2 $ \pmts2 -> do
        SMV.unsafeWith mts3 $ \pmts3 -> do
        SMV.unsafeWith mss $ \pmss -> do
          listVarsArrayHistoriesAlignedTop_u (fromIntegral xmax) (fromIntegral omax) (fromIntegral n) psvv (fromIntegral m) (fromIntegral z) (fromIntegral zrr) ppww phh phhx phhrr phhrrx pmqww1 pmqww2 pmts1 pmts2 pmts3 pmss
      vsqww1' <- SV.unsafeFreeze mqww1 
      vsqww2' <- SV.unsafeFreeze mqww2
      vsts1' <- SV.unsafeFreeze mts1
      vsts2' <- SV.unsafeFreeze mts2
      vsts3' <- SV.unsafeFreeze mts3
      vsss' <- SV.unsafeFreeze mss
      return (SV.take (fromIntegral t) vsqww1',SV.take (fromIntegral t) vsqww2',SV.take (fromIntegral t) vsts1',SV.take (fromIntegral t) vsts2',SV.take (fromIntegral t) vsts3', toInteger (vsss' SV.! 0))
    !qq = V.zip (V.zip3 (SV.convert (SV.unsafeCast vsts1)) (SV.convert (SV.unsafeCast vsts2)) (V.map fromIntegral (SV.convert vsts3))) (V.map (\(p1,p2) -> Set.fromList [vhh V.! (fromIntegral p1), vhh V.! (fromIntegral p2)]) (V.zip (SV.convert vsqww1) (SV.convert vsqww2)))
    qqll = Set.toList
    llvv = V.fromList

parametersSetVarsHistoryRepasSetSetVarsAlignedTop_1 :: Integer -> Integer -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Maybe (Set.Set (Set.Set Variable))
parametersSetVarsHistoryRepasSetSetVarsAlignedTop_1 xmax omax vv hh hhx hhrr hhrrx
  | xmax <= 0 || omax <= 0 || z <= 0 || zrr <= 0 = Nothing
  | not (vhhx == vhh && vhhx == vhh && vhhrr == vhh && vhhrrx == vhh) = Nothing
  | not (vv `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | otherwise = Just $ parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_1 xmax omax vv hh hhx hhrr hhrrx
  where
    HistoryRepa vhh mvv _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    R.Z R.:. _ R.:. z = R.extent aa
    R.Z R.:. _ R.:. zrr = R.extent aarr

foreign import ccall unsafe "listVarsArrayHistoriesAlignedTop_u_1" listVarsArrayHistoriesAlignedTop_u_1
    :: CLLong -> CLLong -> CLLong -> Ptr CLLong -> CLLong -> CLLong -> CLLong -> Ptr CLLong -> Ptr CShort -> Ptr CDouble -> Ptr CShort -> Ptr CDouble -> Ptr CLLong -> Ptr CLLong -> IO (CLLong)

parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_1 :: Integer -> Integer -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Set.Set (Set.Set Variable) 
parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_1 xmax omax ww hh hhx hhrr hhrrx = qq
  where
    HistoryRepa vhh mvv svv aa = hh
    HistogramRepaRed _ _ _ laax = hhx
    HistoryRepa _ _ _ aarr = hhrr
    HistogramRepaRed _ _ _ laarrx = hhrrx
    R.Z R.:. (!n) R.:. (!z) = R.extent aa
    R.Z R.:. _ R.:. (!zrr) = R.extent aarr
    !vww = llvv $ qqll ww 
    !m = V.length vww
    !pww = V.map (mvv Map.!) vww 
    !vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CShort
    !vshhx = SV.unsafeCast (UV.convert (UV.concat (V.toList laax))) :: SV.Vector CDouble
    !vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CShort
    !vshhrrx = SV.unsafeCast (UV.convert (UV.concat (V.toList laarrx))) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vspww = SV.unsafeCast (UV.convert pww) :: SV.Vector CLLong
    (!vsqww1,!vsqww2) = unsafePerformIO $ do
      let vsqww1 = SV.replicate (fromIntegral omax) 0
      let vsqww2 = SV.replicate (fromIntegral omax) 0
      mqww1 <- SV.unsafeThaw vsqww1
      mqww2 <- SV.unsafeThaw vsqww2
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vspww $ \ppww -> do
        SV.unsafeWith vshh $ \phh -> do
        SV.unsafeWith vshhx $ \phhx -> do
        SV.unsafeWith vshhrr $ \phhrr -> do
        SV.unsafeWith vshhrrx $ \phhrrx -> do
        SMV.unsafeWith mqww1 $ \pmqww1 -> do
        SMV.unsafeWith mqww2 $ \pmqww2 -> do
          listVarsArrayHistoriesAlignedTop_u_1 (fromIntegral xmax) (fromIntegral omax) (fromIntegral n) psvv (fromIntegral m) (fromIntegral z) (fromIntegral zrr) ppww phh phhx phhrr phhrrx pmqww1 pmqww2
      vsqww1' <- SV.unsafeFreeze mqww1 
      vsqww2' <- SV.unsafeFreeze mqww2
      return (SV.take (fromIntegral t) vsqww1',SV.take (fromIntegral t) vsqww2')
    !qq = Set.fromList [Set.fromList [vhh V.! (fromIntegral p1), vhh V.! (fromIntegral p2)] | (p1,p2) <- zip (SV.toList vsqww1) (SV.toList vsqww2)]
    qqll = Set.toList
    llvv = V.fromList

parametersSetVarsHistoryRepasSetSetVarsAlignedTop_2 :: Integer -> Integer -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Maybe (Set.Set (Set.Set Variable,Double,Double,Integer),Integer)
parametersSetVarsHistoryRepasSetSetVarsAlignedTop_2 xmax omax vv hh hhx hhrr hhrrx
  | xmax <= 0 || omax <= 0 || z <= 0 || zrr <= 0 = Nothing
  | not (vhhx == vhh && vhhx == vhh && vhhrr == vhh && vhhrrx == vhh) = Nothing
  | not (vv `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | otherwise = Just $ parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_2 xmax omax vv hh hhx hhrr hhrrx
  where
    HistoryRepa vhh mvv _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    R.Z R.:. _ R.:. z = R.extent aa
    R.Z R.:. _ R.:. zrr = R.extent aarr

parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_2 :: Integer -> Integer -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> (Set.Set (Set.Set Variable,Double,Double,Integer),Integer) 
parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_2 xmax omax ww hh hhx hhrr hhrrx = (qq,s)
  where
    HistoryRepa vhh mvv svv aa = hh
    HistogramRepaRed _ _ _ laax = hhx
    HistoryRepa _ _ _ aarr = hhrr
    HistogramRepaRed _ _ _ laarrx = hhrrx
    R.Z R.:. (!n) R.:. (!z) = R.extent aa
    R.Z R.:. _ R.:. (!zrr) = R.extent aarr
    !vww = llvv $ qqll ww 
    !m = V.length vww
    !pww = V.map (mvv Map.!) vww 
    !vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CShort
    !vshhx = SV.unsafeCast (UV.convert (UV.concat (V.toList laax))) :: SV.Vector CDouble
    !vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CShort
    !vshhrrx = SV.unsafeCast (UV.convert (UV.concat (V.toList laarrx))) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vspww = SV.unsafeCast (UV.convert pww) :: SV.Vector CLLong
    (!vsqww1,!vsqww2,!vsts1,!vsts2,!vsts3,!s) = unsafePerformIO $ do
      let vsqww1 = SV.replicate (fromIntegral omax) 0
      let vsqww2 = SV.replicate (fromIntegral omax) 0
      let vsts1 = SV.replicate (fromIntegral omax) 0
      let vsts2 = SV.replicate (fromIntegral omax) 0
      let vsts3 = SV.replicate (fromIntegral omax) 0
      let vsss = SV.replicate 1 0
      mqww1 <- SV.unsafeThaw vsqww1
      mqww2 <- SV.unsafeThaw vsqww2
      mts1 <- SV.unsafeThaw vsts1
      mts2 <- SV.unsafeThaw vsts2
      mts3 <- SV.unsafeThaw vsts3
      mss <- SV.unsafeThaw vsss
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vspww $ \ppww -> do
        SV.unsafeWith vshh $ \phh -> do
        SV.unsafeWith vshhx $ \phhx -> do
        SV.unsafeWith vshhrr $ \phhrr -> do
        SV.unsafeWith vshhrrx $ \phhrrx -> do
        SMV.unsafeWith mqww1 $ \pmqww1 -> do
        SMV.unsafeWith mqww2 $ \pmqww2 -> do
        SMV.unsafeWith mts1 $ \pmts1 -> do
        SMV.unsafeWith mts2 $ \pmts2 -> do
        SMV.unsafeWith mts3 $ \pmts3 -> do
        SMV.unsafeWith mss $ \pmss -> do
          listVarsArrayHistoriesAlignedTop_u (fromIntegral xmax) (fromIntegral omax) (fromIntegral n) psvv (fromIntegral m) (fromIntegral z) (fromIntegral zrr) ppww phh phhx phhrr phhrrx pmqww1 pmqww2 pmts1 pmts2 pmts3 pmss
      vsqww1' <- SV.unsafeFreeze mqww1 
      vsqww2' <- SV.unsafeFreeze mqww2
      vsts1' <- SV.unsafeFreeze mts1
      vsts2' <- SV.unsafeFreeze mts2
      vsts3' <- SV.unsafeFreeze mts3
      vsss' <- SV.unsafeFreeze mss
      return (SV.take (fromIntegral t) vsqww1',SV.take (fromIntegral t) vsqww2',SV.take (fromIntegral t) vsts1',SV.take (fromIntegral t) vsts2',SV.take (fromIntegral t) vsts3', toInteger (vsss' SV.! 0))
    !qq = Set.fromList [(Set.fromList [vhh V.! (fromIntegral p1), vhh V.! (fromIntegral p2)], ts1, ts2, (fromIntegral ts3)) | 
      (p1,p2,ts1,ts2,ts3) <- zip5 (SV.toList vsqww1) (SV.toList vsqww2) (SV.toList (SV.unsafeCast vsts1)) (SV.toList (SV.unsafeCast vsts2)) (SV.toList vsts3)]
    qqll = Set.toList
    llvv = V.fromList

parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop :: Integer -> Integer -> Set.Set Variable -> Set.Set (Set.Set Variable) -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Maybe (V.Vector ((Double,Double,Integer),Set.Set Variable),Integer)
parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop xmax omax vv dd hh hhx hhrr hhrrx
  | xmax <= 0 || omax <= 0 || z <= 0 || zrr <= 0 = Nothing
  | not (vhhx == vhh && vhhx == vhh && vhhrr == vhh && vhhrrx == vhh) = Nothing
  | Set.size vv == 0 = Nothing
  | Set.size dd == 0 = Nothing
  | not (vv `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | not (setSetsUnion dd `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | Set.size (Set.map Set.size dd) /= 1 = Nothing
  | otherwise = Just $ parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u xmax omax vv vdd hh hhx hhrr hhrrx
  where
    HistoryRepa vhh mvv _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    R.Z R.:. _ R.:. z = R.extent aa
    R.Z R.:. _ R.:. zrr = R.extent aarr
    vdd = qqvv dd
    qqvv = V.fromList . Set.toList

foreign import ccall unsafe "listVarsListTuplesArrayHistoriesAlignedTop_u" listVarsListTuplesArrayHistoriesAlignedTop_u
  :: CLLong -> CLLong -> CLLong -> CLLong -> Ptr CLLong -> CLLong -> CLLong -> CLLong -> 
    CLLong -> CLLong -> Ptr CLLong -> Ptr CLLong -> Ptr CShort -> Ptr CDouble -> Ptr CShort -> Ptr CDouble -> 
    Ptr CLLong -> Ptr CLLong -> Ptr CDouble -> Ptr CDouble -> Ptr CLLong -> Ptr CLLong -> IO (CLLong)

parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u :: Integer -> Integer -> Set.Set Variable -> V.Vector (Set.Set Variable) -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> (V.Vector ((Double,Double,Integer),Set.Set Variable),Integer) 
parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u xmax omax ww vdd hh hhx hhrr hhrrx = (qq,s)
  where
    HistoryRepa vhh mvv svv aa = hh
    HistogramRepaRed _ _ _ laax = hhx
    HistoryRepa _ _ _ aarr = hhrr
    HistogramRepaRed _ _ _ laarrx = hhrrx
    R.Z R.:. (!n) R.:. (!z) = R.extent aa
    R.Z R.:. _ R.:. (!zrr) = R.extent aarr
    !vww = qqvv ww 
    !m = V.length vww
    !d = V.length vdd
    !e = Set.size (V.head vdd)
    !pww = V.map (mvv Map.!) vww
    !pdd = V.map (mvv Map.!) (V.concat (V.toList (V.map qqvv vdd)))
    !vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CShort
    !vshhx = SV.unsafeCast (UV.convert (UV.concat (V.toList laax))) :: SV.Vector CDouble
    !vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CShort
    !vshhrrx = SV.unsafeCast (UV.convert (UV.concat (V.toList laarrx))) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vspww = SV.unsafeCast (UV.convert pww) :: SV.Vector CLLong
    !vspdd = SV.unsafeCast (UV.convert pdd) :: SV.Vector CLLong
    (!vsqww1,!vsqww2,!vsts1,!vsts2,!vsts3,!s) = unsafePerformIO $ do
      let vsqww1 = SV.replicate (fromIntegral omax) 0
      let vsqww2 = SV.replicate (fromIntegral omax) 0
      let vsts1 = SV.replicate (fromIntegral omax) 0
      let vsts2 = SV.replicate (fromIntegral omax) 0
      let vsts3 = SV.replicate (fromIntegral omax) 0
      let vsss = SV.replicate 1 0
      mqww1 <- SV.unsafeThaw vsqww1
      mqww2 <- SV.unsafeThaw vsqww2
      mts1 <- SV.unsafeThaw vsts1
      mts2 <- SV.unsafeThaw vsts2
      mts3 <- SV.unsafeThaw vsts3
      mss <- SV.unsafeThaw vsss
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vspww $ \ppww -> do
        SV.unsafeWith vspdd $ \ppdd -> do
        SV.unsafeWith vshh $ \phh -> do
        SV.unsafeWith vshhx $ \phhx -> do
        SV.unsafeWith vshhrr $ \phhrr -> do
        SV.unsafeWith vshhrrx $ \phhrrx -> do
        SMV.unsafeWith mqww1 $ \pmqww1 -> do
        SMV.unsafeWith mqww2 $ \pmqww2 -> do
        SMV.unsafeWith mts1 $ \pmts1 -> do
        SMV.unsafeWith mts2 $ \pmts2 -> do
        SMV.unsafeWith mts3 $ \pmts3 -> do
        SMV.unsafeWith mss $ \pmss -> do
          listVarsListTuplesArrayHistoriesAlignedTop_u 0 (fromIntegral xmax) (fromIntegral omax) 
            (fromIntegral n) psvv (fromIntegral m) (fromIntegral d) (fromIntegral e) 
            (fromIntegral z) (fromIntegral zrr) ppww ppdd phh phhx phhrr phhrrx pmqww1 pmqww2 pmts1 pmts2 pmts3 pmss
      vsqww1' <- SV.unsafeFreeze mqww1 
      vsqww2' <- SV.unsafeFreeze mqww2
      vsts1' <- SV.unsafeFreeze mts1
      vsts2' <- SV.unsafeFreeze mts2
      vsts3' <- SV.unsafeFreeze mts3
      vsss' <- SV.unsafeFreeze mss
      return (SV.take (fromIntegral t) vsqww1',SV.take (fromIntegral t) vsqww2',
        SV.take (fromIntegral t) vsts1',SV.take (fromIntegral t) vsts2',SV.take (fromIntegral t) vsts3', 
        toInteger (vsss' SV.! 0))
    !qq = V.zip (V.zip3 (SV.convert (SV.unsafeCast vsts1)) (SV.convert (SV.unsafeCast vsts2)) (V.map fromIntegral (SV.convert vsts3))) (V.map (\(p1,p2) -> Set.insert (vww V.! (fromIntegral p1)) (vdd V.! (fromIntegral p2))) (V.zip (SV.convert vsqww1) (SV.convert vsqww2)))
    qqvv = V.fromList . Set.toList

parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_1 :: Integer -> Integer -> Set.Set Variable -> Set.Set (Set.Set Variable) -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Maybe (Set.Set (Set.Set Variable,Double,Double,Integer),Integer)
parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_1 xmax omax vv dd hh hhx hhrr hhrrx
  | xmax <= 0 || omax <= 0 || z <= 0 || zrr <= 0 = Nothing
  | not (vhhx == vhh && vhhx == vhh && vhhrr == vhh && vhhrrx == vhh) = Nothing
  | Set.size vv == 0 = Nothing
  | Set.size dd == 0 = Nothing
  | not (vv `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | not (setSetsUnion dd `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | Set.size (Set.map Set.size dd) /= 1 = Nothing
  | otherwise = Just $ parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u_1 xmax omax vv dd hh hhx hhrr hhrrx
  where
    HistoryRepa vhh mvv _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    R.Z R.:. _ R.:. z = R.extent aa
    R.Z R.:. _ R.:. zrr = R.extent aarr

foreign import ccall unsafe "listVarsListTuplesArrayHistoriesAlignedTop_u_1" listVarsListTuplesArrayHistoriesAlignedTop_u_1
  :: CLLong -> CLLong -> CLLong -> Ptr CLLong -> CLLong -> CLLong -> CLLong -> 
    CLLong -> CLLong -> Ptr CLLong -> Ptr CLLong -> Ptr CShort -> Ptr CDouble -> Ptr CShort -> Ptr CDouble -> 
    Ptr CLLong -> Ptr CLLong -> Ptr CDouble -> Ptr CDouble -> Ptr CLLong -> Ptr CLLong -> IO (CLLong)

parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u_1 :: Integer -> Integer -> Set.Set Variable -> Set.Set (Set.Set Variable) -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> (Set.Set (Set.Set Variable,Double,Double,Integer),Integer) 
parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u_1 xmax omax ww dd hh hhx hhrr hhrrx = (qq,s)
  where
    HistoryRepa vhh mvv svv aa = hh
    HistogramRepaRed _ _ _ laax = hhx
    HistoryRepa _ _ _ aarr = hhrr
    HistogramRepaRed _ _ _ laarrx = hhrrx
    R.Z R.:. (!n) R.:. (!z) = R.extent aa
    R.Z R.:. _ R.:. (!zrr) = R.extent aarr
    !vww = qqvv ww 
    !m = V.length vww
    !vdd = qqvv dd 
    !d = V.length vdd
    !e = Set.size (V.head vdd)
    !pww = V.map (mvv Map.!) vww
    !pdd = V.map (mvv Map.!) (V.concat (V.toList (V.map qqvv vdd)))
    !vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CShort
    !vshhx = SV.unsafeCast (UV.convert (UV.concat (V.toList laax))) :: SV.Vector CDouble
    !vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CShort
    !vshhrrx = SV.unsafeCast (UV.convert (UV.concat (V.toList laarrx))) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vspww = SV.unsafeCast (UV.convert pww) :: SV.Vector CLLong
    !vspdd = SV.unsafeCast (UV.convert pdd) :: SV.Vector CLLong
    (!vsqww1,!vsqww2,!vsts1,!vsts2,!vsts3,!s) = unsafePerformIO $ do
      let vsqww1 = SV.replicate (fromIntegral omax) 0
      let vsqww2 = SV.replicate (fromIntegral omax) 0
      let vsts1 = SV.replicate (fromIntegral omax) 0
      let vsts2 = SV.replicate (fromIntegral omax) 0
      let vsts3 = SV.replicate (fromIntegral omax) 0
      let vsss = SV.replicate 1 0
      mqww1 <- SV.unsafeThaw vsqww1
      mqww2 <- SV.unsafeThaw vsqww2
      mts1 <- SV.unsafeThaw vsts1
      mts2 <- SV.unsafeThaw vsts2
      mts3 <- SV.unsafeThaw vsts3
      mss <- SV.unsafeThaw vsss
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vspww $ \ppww -> do
        SV.unsafeWith vspdd $ \ppdd -> do
        SV.unsafeWith vshh $ \phh -> do
        SV.unsafeWith vshhx $ \phhx -> do
        SV.unsafeWith vshhrr $ \phhrr -> do
        SV.unsafeWith vshhrrx $ \phhrrx -> do
        SMV.unsafeWith mqww1 $ \pmqww1 -> do
        SMV.unsafeWith mqww2 $ \pmqww2 -> do
        SMV.unsafeWith mts1 $ \pmts1 -> do
        SMV.unsafeWith mts2 $ \pmts2 -> do
        SMV.unsafeWith mts3 $ \pmts3 -> do
        SMV.unsafeWith mss $ \pmss -> do
          listVarsListTuplesArrayHistoriesAlignedTop_u_1 (fromIntegral xmax) (fromIntegral omax) 
            (fromIntegral n) psvv (fromIntegral m) (fromIntegral d) (fromIntegral e) 
            (fromIntegral z) (fromIntegral zrr) ppww ppdd phh phhx phhrr phhrrx pmqww1 pmqww2 pmts1 pmts2 pmts3 pmss
      vsqww1' <- SV.unsafeFreeze mqww1 
      vsqww2' <- SV.unsafeFreeze mqww2
      vsts1' <- SV.unsafeFreeze mts1
      vsts2' <- SV.unsafeFreeze mts2
      vsts3' <- SV.unsafeFreeze mts3
      vsss' <- SV.unsafeFreeze mss
      return (SV.take (fromIntegral t) vsqww1',SV.take (fromIntegral t) vsqww2',
        SV.take (fromIntegral t) vsts1',SV.take (fromIntegral t) vsts2',SV.take (fromIntegral t) vsts3', 
        toInteger (vsss' SV.! 0))
    !qq = Set.fromList [(Set.insert (vww V.! (fromIntegral p1)) (vdd V.! (fromIntegral p2)), 
      ts1, ts2, (fromIntegral ts3)) | (p1,p2,ts1,ts2,ts3) <- zip5 (SV.toList vsqww1) (SV.toList vsqww2) 
        (SV.toList (SV.unsafeCast vsts1)) (SV.toList (SV.unsafeCast vsts2)) (SV.toList vsts3)]
    qqvv = V.fromList . Set.toList

parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u_2 :: Integer -> Integer -> Set.Set Variable -> V.Vector (Set.Set Variable) -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> (V.Vector ((Double,Double,Integer),Set.Set Variable),Integer) 
parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u_2 xmax omax ww vdd hh hhx hhrr hhrrx = (qq,s)
  where
    HistoryRepa vhh mvv svv aa = hh
    HistogramRepaRed _ _ _ laax = hhx
    HistoryRepa _ _ _ aarr = hhrr
    HistogramRepaRed _ _ _ laarrx = hhrrx
    R.Z R.:. (!n) R.:. (!z) = R.extent aa
    R.Z R.:. _ R.:. (!zrr) = R.extent aarr
    !vww = qqvv ww 
    !m = V.length vww
    !d = V.length vdd
    !e = Set.size (V.head vdd)
    !pww = V.map (mvv Map.!) vww
    !pdd = V.map (mvv Map.!) (V.concat (V.toList (V.map qqvv vdd)))
    !vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CShort
    !vshhx = SV.unsafeCast (UV.convert (UV.concat (V.toList laax))) :: SV.Vector CDouble
    !vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CShort
    !vshhrrx = SV.unsafeCast (UV.convert (UV.concat (V.toList laarrx))) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vspww = SV.unsafeCast (UV.convert pww) :: SV.Vector CLLong
    !vspdd = SV.unsafeCast (UV.convert pdd) :: SV.Vector CLLong
    (!vsqww1,!vsqww2,!vsts1,!vsts2,!vsts3,!s) = unsafePerformIO $ do
      let vsqww1 = SV.replicate (fromIntegral omax) 0
      let vsqww2 = SV.replicate (fromIntegral omax) 0
      let vsts1 = SV.replicate (fromIntegral omax) 0
      let vsts2 = SV.replicate (fromIntegral omax) 0
      let vsts3 = SV.replicate (fromIntegral omax) 0
      let vsss = SV.replicate 1 0
      mqww1 <- SV.unsafeThaw vsqww1
      mqww2 <- SV.unsafeThaw vsqww2
      mts1 <- SV.unsafeThaw vsts1
      mts2 <- SV.unsafeThaw vsts2
      mts3 <- SV.unsafeThaw vsts3
      mss <- SV.unsafeThaw vsss
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vspww $ \ppww -> do
        SV.unsafeWith vspdd $ \ppdd -> do
        SV.unsafeWith vshh $ \phh -> do
        SV.unsafeWith vshhx $ \phhx -> do
        SV.unsafeWith vshhrr $ \phhrr -> do
        SV.unsafeWith vshhrrx $ \phhrrx -> do
        SMV.unsafeWith mqww1 $ \pmqww1 -> do
        SMV.unsafeWith mqww2 $ \pmqww2 -> do
        SMV.unsafeWith mts1 $ \pmts1 -> do
        SMV.unsafeWith mts2 $ \pmts2 -> do
        SMV.unsafeWith mts3 $ \pmts3 -> do
        SMV.unsafeWith mss $ \pmss -> do
          listVarsListTuplesArrayHistoriesAlignedTop_u_1 (fromIntegral xmax) (fromIntegral omax) 
            (fromIntegral n) psvv (fromIntegral m) (fromIntegral d) (fromIntegral e) 
            (fromIntegral z) (fromIntegral zrr) ppww ppdd phh phhx phhrr phhrrx pmqww1 pmqww2 pmts1 pmts2 pmts3 pmss
      vsqww1' <- SV.unsafeFreeze mqww1 
      vsqww2' <- SV.unsafeFreeze mqww2
      vsts1' <- SV.unsafeFreeze mts1
      vsts2' <- SV.unsafeFreeze mts2
      vsts3' <- SV.unsafeFreeze mts3
      vsss' <- SV.unsafeFreeze mss
      return (SV.take (fromIntegral t) vsqww1',SV.take (fromIntegral t) vsqww2',
        SV.take (fromIntegral t) vsts1',SV.take (fromIntegral t) vsts2',SV.take (fromIntegral t) vsts3', 
        toInteger (vsss' SV.! 0))
    !qq = V.zip (V.zip3 (SV.convert (SV.unsafeCast vsts1)) (SV.convert (SV.unsafeCast vsts2)) (V.map fromIntegral (SV.convert vsts3))) (V.map (\(p1,p2) -> Set.insert (vww V.! (fromIntegral p1)) (vdd V.! (fromIntegral p2))) (V.zip (SV.convert vsqww1) (SV.convert vsqww2)))
    qqvv = V.fromList . Set.toList

parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop :: Integer -> Integer -> Set.Set Variable -> Set.Set (Set.Set Variable) -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Maybe (V.Vector ((Double,Double,Integer),Set.Set Variable),Integer)
parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop wmax omax vv dd hh hhx hhrr hhrrx
  | wmax <= 0 || omax <= 0 || z <= 0 || zrr <= 0 = Nothing
  | not (vhhx == vhh && vhhx == vhh && vhhrr == vhh && vhhrrx == vhh) = Nothing
  | Set.size vv == 0 = Nothing
  | Set.size dd == 0 = Nothing
  | not (vv `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | not (setSetsUnion dd `Set.isSubsetOf` (Map.keysSet mvv)) = Nothing
  | Set.size (Set.map Set.size dd) /= 1 = Nothing
  | otherwise = Just $ parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop_u wmax omax vv vdd hh hhx hhrr hhrrx
  where
    HistoryRepa vhh mvv _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    R.Z R.:. _ R.:. z = R.extent aa
    R.Z R.:. _ R.:. zrr = R.extent aarr
    vdd = qqvv dd
    qqvv = V.fromList . Set.toList

parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop_u :: Integer -> Integer -> Set.Set Variable -> V.Vector (Set.Set Variable) -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> (V.Vector ((Double,Double,Integer),Set.Set Variable),Integer) 
parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop_u wmax omax ww vdd hh hhx hhrr hhrrx = (qq,s)
  where
    HistoryRepa vhh mvv svv aa = hh
    HistogramRepaRed _ _ _ laax = hhx
    HistoryRepa _ _ _ aarr = hhrr
    HistogramRepaRed _ _ _ laarrx = hhrrx
    R.Z R.:. (!n) R.:. (!z) = R.extent aa
    R.Z R.:. _ R.:. (!zrr) = R.extent aarr
    !vww = qqvv ww 
    !m = V.length vww
    !d = V.length vdd
    !e = Set.size (V.head vdd)
    !pww = V.map (mvv Map.!) vww
    !pdd = V.map (mvv Map.!) (V.concat (V.toList (V.map qqvv vdd)))
    !vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CShort
    !vshhx = SV.unsafeCast (UV.convert (UV.concat (V.toList laax))) :: SV.Vector CDouble
    !vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CShort
    !vshhrrx = SV.unsafeCast (UV.convert (UV.concat (V.toList laarrx))) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vspww = SV.unsafeCast (UV.convert pww) :: SV.Vector CLLong
    !vspdd = SV.unsafeCast (UV.convert pdd) :: SV.Vector CLLong
    (!vsqww1,!vsqww2,!vsts1,!vsts2,!vsts3,!s) = unsafePerformIO $ do
      let vsqww1 = SV.replicate (fromIntegral omax) 0
      let vsqww2 = SV.replicate (fromIntegral omax) 0
      let vsts1 = SV.replicate (fromIntegral omax) 0
      let vsts2 = SV.replicate (fromIntegral omax) 0
      let vsts3 = SV.replicate (fromIntegral omax) 0
      let vsss = SV.replicate 1 0
      mqww1 <- SV.unsafeThaw vsqww1
      mqww2 <- SV.unsafeThaw vsqww2
      mts1 <- SV.unsafeThaw vsts1
      mts2 <- SV.unsafeThaw vsts2
      mts3 <- SV.unsafeThaw vsts3
      mss <- SV.unsafeThaw vsss
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vspww $ \ppww -> do
        SV.unsafeWith vspdd $ \ppdd -> do
        SV.unsafeWith vshh $ \phh -> do
        SV.unsafeWith vshhx $ \phhx -> do
        SV.unsafeWith vshhrr $ \phhrr -> do
        SV.unsafeWith vshhrrx $ \phhrrx -> do
        SMV.unsafeWith mqww1 $ \pmqww1 -> do
        SMV.unsafeWith mqww2 $ \pmqww2 -> do
        SMV.unsafeWith mts1 $ \pmts1 -> do
        SMV.unsafeWith mts2 $ \pmts2 -> do
        SMV.unsafeWith mts3 $ \pmts3 -> do
        SMV.unsafeWith mss $ \pmss -> do
          listVarsListTuplesArrayHistoriesAlignedTop_u 1 (fromIntegral wmax) (fromIntegral omax) 
            (fromIntegral n) psvv (fromIntegral m) (fromIntegral d) (fromIntegral e) 
            (fromIntegral z) (fromIntegral zrr) ppww ppdd phh phhx phhrr phhrrx pmqww1 pmqww2 pmts1 pmts2 pmts3 pmss
      vsqww1' <- SV.unsafeFreeze mqww1 
      vsqww2' <- SV.unsafeFreeze mqww2
      vsts1' <- SV.unsafeFreeze mts1
      vsts2' <- SV.unsafeFreeze mts2
      vsts3' <- SV.unsafeFreeze mts3
      vsss' <- SV.unsafeFreeze mss
      return (SV.take (fromIntegral t) vsqww1',SV.take (fromIntegral t) vsqww2',
        SV.take (fromIntegral t) vsts1',SV.take (fromIntegral t) vsts2',SV.take (fromIntegral t) vsts3', 
        toInteger (vsss' SV.! 0))
    !qq = V.zip (V.zip3 (SV.convert (SV.unsafeCast vsts1)) (SV.convert (SV.unsafeCast vsts2)) (V.map fromIntegral (SV.convert vsts3))) (V.map (\(p1,p2) -> Set.insert (vww V.! (fromIntegral p1)) (vdd V.! (fromIntegral p2))) (V.zip (SV.convert vsqww1) (SV.convert vsqww2)))
    qqvv = V.fromList . Set.toList

foreign import ccall unsafe "listVarsListTuplesArrayHistoriesAlignedExcludeHiddenTop_u" listVarsListTuplesArrayHistoriesAlignedExcludeHiddenTop_u
  :: CLLong -> CLLong -> CLLong -> CLLong -> Ptr CLLong -> CLLong -> CLLong -> CLLong -> 
    CLLong -> CLLong -> CLLong -> Ptr CLLong -> Ptr CLLong -> Ptr CLLong -> Ptr CLLong -> Ptr CShort -> Ptr CDouble -> Ptr CShort -> Ptr CDouble -> 
    Ptr CLLong -> Ptr CLLong -> Ptr CDouble -> Ptr CDouble -> Ptr CLLong -> Ptr CLLong -> IO (CLLong)

parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedExcludeHiddenDenseTop_u :: Integer -> Integer -> Set.Set (Variable,Variable) -> Set.Set Variable -> V.Vector (Set.Set Variable) -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> (V.Vector ((Double,Double,Integer),Set.Set Variable),Integer) 
parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedExcludeHiddenDenseTop_u wmax omax cc ww vdd hh hhx hhrr hhrrx = (qq,s)
  where
    HistoryRepa vhh mvv svv aa = hh
    HistogramRepaRed _ _ _ laax = hhx
    HistoryRepa _ _ _ aarr = hhrr
    HistogramRepaRed _ _ _ laarrx = hhrrx
    R.Z R.:. (!n) R.:. (!z) = R.extent aa
    R.Z R.:. _ R.:. (!zrr) = R.extent aarr
    !vcc = qqvv cc
    !ccl = V.length vcc
    !pccd = V.map (\r -> mvv Map.! (fst r)) vcc
    !pccu = V.map (\r -> mvv Map.! (snd r)) vcc
    !vww = qqvv ww 
    !m = V.length vww
    !d = V.length vdd
    !e = Set.size (V.head vdd)
    !pww = V.map (mvv Map.!) vww
    !pdd = V.map (mvv Map.!) (V.concat (V.toList (V.map qqvv vdd)))
    !vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CShort
    !vshhx = SV.unsafeCast (UV.convert (UV.concat (V.toList laax))) :: SV.Vector CDouble
    !vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CShort
    !vshhrrx = SV.unsafeCast (UV.convert (UV.concat (V.toList laarrx))) :: SV.Vector CDouble
    !vssvv = SV.unsafeCast (UV.convert svv) :: SV.Vector CLLong
    !vspccd = SV.unsafeCast (UV.convert pccd) :: SV.Vector CLLong
    !vspccu = SV.unsafeCast (UV.convert pccu) :: SV.Vector CLLong
    !vspww = SV.unsafeCast (UV.convert pww) :: SV.Vector CLLong
    !vspdd = SV.unsafeCast (UV.convert pdd) :: SV.Vector CLLong
    (!vsqww1,!vsqww2,!vsts1,!vsts2,!vsts3,!s) = unsafePerformIO $ do
      let vsqww1 = SV.replicate (fromIntegral omax) 0
      let vsqww2 = SV.replicate (fromIntegral omax) 0
      let vsts1 = SV.replicate (fromIntegral omax) 0
      let vsts2 = SV.replicate (fromIntegral omax) 0
      let vsts3 = SV.replicate (fromIntegral omax) 0
      let vsss = SV.replicate 1 0
      mqww1 <- SV.unsafeThaw vsqww1
      mqww2 <- SV.unsafeThaw vsqww2
      mts1 <- SV.unsafeThaw vsts1
      mts2 <- SV.unsafeThaw vsts2
      mts3 <- SV.unsafeThaw vsts3
      mss <- SV.unsafeThaw vsss
      t <- SV.unsafeWith vssvv $ \psvv -> do
        SV.unsafeWith vspccd $ \ppccd -> do
        SV.unsafeWith vspccu $ \ppccu -> do
        SV.unsafeWith vspww $ \ppww -> do
        SV.unsafeWith vspdd $ \ppdd -> do
        SV.unsafeWith vshh $ \phh -> do
        SV.unsafeWith vshhx $ \phhx -> do
        SV.unsafeWith vshhrr $ \phhrr -> do
        SV.unsafeWith vshhrrx $ \phhrrx -> do
        SMV.unsafeWith mqww1 $ \pmqww1 -> do
        SMV.unsafeWith mqww2 $ \pmqww2 -> do
        SMV.unsafeWith mts1 $ \pmts1 -> do
        SMV.unsafeWith mts2 $ \pmts2 -> do
        SMV.unsafeWith mts3 $ \pmts3 -> do
        SMV.unsafeWith mss $ \pmss -> do
          listVarsListTuplesArrayHistoriesAlignedExcludeHiddenTop_u 1 (fromIntegral wmax) (fromIntegral omax) 
            (fromIntegral n) psvv (fromIntegral m) (fromIntegral d) (fromIntegral e) 
            (fromIntegral z) (fromIntegral zrr) (fromIntegral ccl) ppccd ppccu ppww ppdd phh phhx phhrr phhrrx pmqww1 pmqww2 pmts1 pmts2 pmts3 pmss
      vsqww1' <- SV.unsafeFreeze mqww1 
      vsqww2' <- SV.unsafeFreeze mqww2
      vsts1' <- SV.unsafeFreeze mts1
      vsts2' <- SV.unsafeFreeze mts2
      vsts3' <- SV.unsafeFreeze mts3
      vsss' <- SV.unsafeFreeze mss
      return (SV.take (fromIntegral t) vsqww1',SV.take (fromIntegral t) vsqww2',
        SV.take (fromIntegral t) vsts1',SV.take (fromIntegral t) vsts2',SV.take (fromIntegral t) vsts3', 
        toInteger (vsss' SV.! 0))
    !qq = V.zip (V.zip3 (SV.convert (SV.unsafeCast vsts1)) (SV.convert (SV.unsafeCast vsts2)) (V.map fromIntegral (SV.convert vsts3))) (V.map (\(p1,p2) -> Set.insert (vww V.! (fromIntegral p1)) (vdd V.! (fromIntegral p2))) (V.zip (SV.convert vsqww1) (SV.convert vsqww2)))
    qqvv = V.fromList . Set.toList

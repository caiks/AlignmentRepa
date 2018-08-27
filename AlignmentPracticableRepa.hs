{-# LANGUAGE RankNTypes, BangPatterns #-}

module AlignmentPracticableRepa (
  parametersSystemsBuilderTupleRepa,
  parametersSystemsBuilderTupleRepa_1,
  parametersSystemsBuilderTupleRepa_2,
  parametersSystemsBuilderTupleRepa_3,
  parametersSystemsBuilderTupleNoSumlayerRepa,
  parametersSystemsBuilderTupleNoSumlayerRepa_1,
  parametersSystemsBuilderTupleNoSumlayerRepa_2,
  parametersSystemsBuilderTupleNoSumlayerRepa_u,
  parametersSystemsBuilderTupleNoSumlayerRepa_u_1,
  parametersSystemsBuilderTupleNoSumlayerRepa_ui,
  parametersSystemsBuilderTupleNoSumlayerRepa_ui_1,
  parametersSystemsBuilderTupleNoSumlayerRepa_ui_2,
  parametersSystemsBuilderTupleNoSumlayerRepa_ui_3,
  parametersSystemsBuilderTupleNoSumlayerRepa_ui_4,
  parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_u,
  parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_u_1,
  parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_ui,
  parametersSystemsBuilderTupleLevelNoSumlayerRepa_u,
  parametersSystemsBuilderTupleLevelNoSumlayerRepa_u_1,
  parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui,
  parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_1,
  parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_2,
  parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_3,
  parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_u,
  parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_u_1,
  parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_ui,
  parametersSystemsBuilderDerivedVarsHighestRepa,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_u,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_ui,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_1,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_2,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_u,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_u_1,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui_1,
  parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui_2,
  parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_u,
  parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_ui,
  parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_u,
  parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_u_1,
  parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui,
  parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui_1,
  parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui_2,
  parametersSystemsPartitionerRepa,
  parametersSystemsPartitionerRepa_1,
  parametersSystemsPartitionerRepa_2,
  parametersSystemsPartitionerRepa_3,
  parametersSystemsPartitionerRepa_4,
  parametersSystemsPartitionerRepa_5,
  parametersSystemsPartitionerRepa_u,
  parametersSystemsPartitionerRepa_u_1,
  parametersSystemsPartitionerRepa_ui,
  parametersSystemsPartitionerRepa_ui_1,
  parametersSystemsPartitionerRepa_ui_2,
  parametersSystemsPartitionerRepa_ui_3,
  parametersSystemsPartitionerMaxRollByMRepa,
  parametersSystemsPartitionerMaxRollByMRepa_u,
  parametersSystemsPartitionerMaxRollByMRepa_ui,
  parametersRollerRepa,
  parametersRollerMaximumRollRepa,
  parametersRollerMaximumRollExcludedSelfRepa,
  parametersRollerMaximumRollExcludedSelfRepa_1,
  parametersRollerMaximumRollExcludedSelfRepa_2,
  parametersRollerMaximumRollExcludedSelfRepa_3,
  parametersRollerMaximumRollExcludedSelfRepa_i,
  parametersRollerMaximumRollExcludedSelfRepa_i_1,
  parametersRollerMaximumRollExcludedSelfRepa_i_2,
  parametersSystemsLayererHighestRepa,
  parametersSystemsLayererMaximumRollHighestRepa,
  parametersSystemsLayererMaximumRollExcludedSelfHighestRepa,
  parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_1,
  parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_2,
  parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u,
  parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u_1,
  parametersSystemsLayererMaxRollByMExcludedSelfHighestRepa,
  parametersSystemsLayererMaxRollByMExcludedSelfHighestRepa_u,
  parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u,
  parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u_1,
  parametersSystemsLayererLevelMaxRollByMExcludedSelfHighestRepa_u,
  parametersSystemsDecomperHighestRepa,
  parametersSystemsDecomperHighestFmaxRepa,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa_1,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa_2,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_1,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_2,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_3,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_4,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxBatchRepa,
  parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxRepa,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyRepa,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelRepa,
  parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelRepa,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelGoodnessRepa,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelRepa,
  parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelRepa,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelGoodnessRepa,
  systemsDecompFudsHistoryRepasAlignmentContentShuffleSummation_u,
  systemsDecompFudsHistoryRepasTreeAlignmentContentShuffleSummation_u,
  systemsDecompFudsHistoryRepasAlgnDensPerSizesStripped_u,
  parametersSystemsBuilderLabelTupleRepa
)
where
import Data.List as List
import qualified Data.Set as Set
import qualified Data.Map as Map
import qualified Data.IntMap as IntMap
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Data.Vector.Storable as SV
import Data.Array.Repa as R
import Foreign.C.Types
import AlignmentUtil
import Alignment
import AlignmentRandom
import AlignmentSubstrate
import AlignmentApprox
import AlignmentRepaVShape
import AlignmentRepa
import AlignmentRandomRepa
import GHC.Real

data MaxRollType = MaximumRoll | MaxRollByM
                     deriving (Eq, Ord, Read, Show)

repaRounding :: Double 
repaRounding = 1e-6

parametersSystemsBuilderTupleRepa :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleRepa xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | xmax < 0 || omax < 0 || mmax < 1 || bmax < mmax = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | ff == fudEmpty = 
      Just $ topd (bmax `div` mmax) $ buildb vv (init vv) []
  | fvars ff `subset` vvqq vhh = 
      Just $ topd (bmax `div` mmax) $ buildb (fvars ff `union` vv) (init (fder ff)) []
  | otherwise = Nothing
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = [(((sgl w, (hvempty, hvempty, UV.empty)),0),(0,0,0,0)) | w <- qqll vv]
    buildb ww qq nn = if mm /= [] then buildb ww mm (nn List.++ mm) else (final nn) 
      where
        pp = llqq [jj | (((kk,_),_),_) <- qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ [(((jj, (bbv,ffv,ssv)), a1-b1), (a1-a2-b1+b2, -l, -b1+b2, -u)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, let l =  sumlayer ff jj, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    final = List.filter (\(((kk,_),_),_) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))
    sumlayer ff kk = sum [layer ff (sgl w) | w <- qqll kk]
    layer = fudsSetVarsLayer
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderTupleRepa_1 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [(Set.Set Variable, HistogramRepa, HistogramRepa)]
parametersSystemsBuilderTupleRepa_1 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | xmax < 0 || omax < 0 || mmax < 1 || bmax < mmax = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | ff == fudEmpty = 
      Just $ topd (bmax `div` mmax) $ buildb vv (init vv) []
  | fvars ff `subset` vvqq vhh = 
      Just $ topd (bmax `div` mmax) $ buildb (fvars ff `union` vv) (init (fder ff)) []
  | otherwise = Nothing
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = [((sgl w, hempty, hempty),(0,0,0,0)) | w <- qqll vv]
    buildb ww qq nn = if mm /= [] then buildb ww mm (nn List.++ mm) else (final nn) 
      where
        pp = llqq [jj | ((kk,_,_),_) <- qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ [((jj, bb, bbrr), (a-b, -l, -b, -u)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, let l =  sumlayer ff jj, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx]
    final = List.filter (\((kk,_,_),_) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))
    sumlayer ff kk = sum [layer ff (sgl w) | w <- qqll kk]
    layer = fudsSetVarsLayer
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderTupleRepa_2 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, HistogramRepa, HistogramRepa),Double)]
parametersSystemsBuilderTupleRepa_2 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | xmax < 0 || omax < 0 || mmax < 1 || bmax < mmax = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | ff == fudEmpty = 
      Just $ topd (bmax `div` mmax) $ buildb vv (init vv) []
  | fvars ff `subset` vvqq vhh = 
      Just $ topd (bmax `div` mmax) $ buildb (fvars ff `union` vv) (init (fder ff)) []
  | otherwise = Nothing
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = [(((sgl w, hempty, hempty),0),(0,0,0,0)) | w <- qqll vv]
    buildb ww qq nn = if mm /= [] then buildb ww mm (nn List.++ mm) else (final nn) 
      where
        pp = llqq [jj | (((kk,_,_),_),_) <- qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ [(((jj, bb, bbrr), a1-b1), (a1-a2-b1+b2, -l, -b1+b2, -u)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, let l =  sumlayer ff jj, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let a1 = sumfacln bb, let a2 = sumfacln bbx, 
          let b1 = sumfacln bbrr, let b2 =sumfacln bbrrx]
    final = List.filter (\(((kk,_,_),_),_) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))
    sumlayer ff kk = sum [layer ff (sgl w) | w <- qqll kk]
    layer = fudsSetVarsLayer
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderTupleRepa_3 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, (HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleRepa_3 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | xmax < 0 || omax < 0 || mmax < 1 || bmax < mmax = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | ff == fudEmpty = 
      Just $ topd (bmax `div` mmax) $ buildb vv (init vv) []
  | fvars ff `subset` vvqq vhh = 
      Just $ topd (bmax `div` mmax) $ buildb (fvars ff `union` vv) (init (fder ff)) []
  | otherwise = Nothing
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = [(((sgl w, (hvempty, UV.empty)),0),(0,0,0,0)) | w <- qqll vv]
    buildb ww qq nn = if mm /= [] then buildb ww mm (nn List.++ mm) else (final nn) 
      where
        pp = llqq [jj | (((kk,_),_),_) <- qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ [(((jj, rr), a1-b1), (a1-a2-b1+b2, -l, -b1+b2, -u)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, let l =  sumlayer ff jj, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let a1 = sumfacln bb, let a2 = sumfacln bbx, 
          let b1 = sumfacln bbrr, let b2 =sumfacln bbrrx,
          let rr = (vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], UV.fromListN 4 [a1,a2,b1,b2])]
    final = List.filter (\(((kk,_),_),_) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))
    sumlayer ff kk = sum [layer ff (sgl w) | w <- qqll kk]
    layer = fudsSetVarsLayer
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderTupleNoSumlayerRepa :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleNoSumlayerRepa xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | xmax < 0 || omax < 0 || mmax < 1 || bmax < mmax = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | otherwise = Just $ buildfftup xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    buildfftup = parametersSystemsBuilderTupleNoSumlayerRepa_u
    uvars = systemsVars
    subset = Set.isSubsetOf
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderTupleNoSumlayerRepa_1 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleNoSumlayerRepa_1 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | xmax < 0 || omax < 0 || mmax < 1 || bmax < mmax = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | ff == fudEmpty = 
      Just $ topd (bmax `div` mmax) $ buildb vv (init vv) []
  | fvars ff `subset` vvqq vhh = 
      Just $ topd (bmax `div` mmax) $ buildb (fvars ff `union` vv) (init (fder ff)) []
  | otherwise = Nothing
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = [(((sgl w, (hvempty, hvempty, UV.empty)),0),(0,0,0)) | w <- qqll vv]
    buildb ww qq nn = if mm /= [] then buildb ww mm (nn List.++ mm) else (final nn) 
      where
        pp = llqq [jj | (((kk,_),_),_) <- qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ [(((jj, (bbv,ffv,ssv)), a1-b1), (a1-a2-b1+b2, -b1+b2, -u)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    final = List.filter (\(((kk,_),_),_) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))
    sumlayer ff kk = sum [layer ff (sgl w) | w <- qqll kk]
    layer = fudsSetVarsLayer
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderTupleNoSumlayerRepa_2 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleNoSumlayerRepa_2 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | xmax < 0 || omax < 0 || mmax < 1 || bmax < mmax = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | ff == fudEmpty = 
      Just $ V.toList $ topd (bmax `div` mmax) $ buildb vv (init vv) V.empty
  | fvars ff `subset` vvqq vhh = 
      Just $ V.toList $ topd (bmax `div` mmax) $ buildb (fvars ff `union` vv) (init (fder ff)) V.empty
  | otherwise = Nothing
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn = if (not (V.null mm)) then buildb ww mm (nn V.++ mm) else (final nn) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    sumlayer ff kk = sum [layer ff (sgl w) | w <- qqll kk]
    layer = fudsSetVarsLayer
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderTupleNoSumlayerRepa_u :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleNoSumlayerRepa_u xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx =
    fst $ buildfftup xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  where
    buildfftup = parametersSystemsBuilderTupleNoSumlayerRepa_ui

parametersSystemsBuilderTupleNoSumlayerRepa_u_1 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleNoSumlayerRepa_u_1 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | ff == fudEmpty = V.toList $ topd (bmax `div` mmax) $ buildb vv (init vv) V.empty
  | otherwise = V.toList $ topd (bmax `div` mmax) $ buildb (fvars ff `union` vv) (init (fder ff)) V.empty
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn = if (not (V.null mm)) then buildb ww mm (nn V.++ mm) else (final nn) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_u :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_u xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx =
    fst $ buildfftup xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  where
    buildfftup = parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_ui 

parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_u_1 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_u_1 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | vv' == Set.empty = []
  | ff == fudEmpty = V.toList $ topd (bmax `div` mmax) $ buildb vv' (init vv') V.empty
  | otherwise = V.toList $ topd (bmax `div` mmax) $ buildb (fvars ff `union` vv') (init (fder ff)) V.empty
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    vv' = meff hhx vv
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn = if (not (V.null mm)) then buildb ww mm (nn V.++ mm) else (final nn) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1)
    meff hhx vv = Set.fromList [v | v <- qqll vv, let i = hhxv hhx Map.! v, 
      length (List.filter (/=0) (UV.toList (hhxi hhx V.! i))) > 1]
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    hhxv = histogramRepaRedsMapVarInt
    hhxi = histogramRepaRedsVectorArray
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleNoSumlayerRepa_ui :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleNoSumlayerRepa_ui xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (res (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (res (topd (bmax `div` mmax) x1), s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    (xc,sc) = cross xmax omax vv hh hhx hhrr hhrrx
    yy = fvars ff `union` vv
    (xa,sa) = append xmax omax yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x0,s0) = buildb vv xc xc sc
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append xmax omax ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    res xx = [((jj, (bbv,ffv,ssv)), a1-b1) | jj <- vvll xx,
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,_,b1,_] = UV.toList ssv]
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    cross = parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    union = Set.union
    sgl = Set.singleton
    vvll = V.toList
    qqvv = V.fromList . Set.toList

parametersSystemsBuilderTupleNoSumlayerRepa_ui_1 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleNoSumlayerRepa_ui_1 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (V.toList (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (V.toList (topd (bmax `div` mmax) x1), s1)
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    (x0,s0) = buildb vv (init vv) V.empty 0
    (x1,s1) = buildb (fvars ff `union` vv) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn s2 = 
        if (not (V.null mm)) then buildb ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
        mm = top omax x2
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce_3
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleNoSumlayerRepa_ui_2 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleNoSumlayerRepa_ui_2 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (V.toList (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (V.toList (topd (bmax `div` mmax) x1), s1)
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CLLong
    (x0,s0) = buildb vv (init vv) V.empty 0
    (x1,s1) = buildb (fvars ff `union` vv) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn s2 = 
        if (not (V.null mm)) then buildb ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
        mm = top omax x2
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleNoSumlayerRepa_ui_3 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleNoSumlayerRepa_ui_3 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (V.toList (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (V.toList (topd (bmax `div` mmax) x1), s1)
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CLLong
    xc = initc vv
    (x0,s0) = buildb vv xc xc (toInteger (Set.size vv * (Set.size vv - 1) `div` 2))
    (x1,s1) = buildb (fvars ff `union` vv) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    initc vv = 
      let pp = cross xmax omax vv hh hhx hhrr hhrrx in
      V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |              
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    buildb ww qq nn s2 = 
        if (not (V.null mm)) then buildb ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
        mm = top omax x2
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    cross = parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_1
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleNoSumlayerRepa_ui_4 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleNoSumlayerRepa_ui_4 xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (t0, s0) 
  | otherwise = (t1, s1)
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CLLong
    (xc,sc) = cross xmax omax vv hh hhx hhrr hhrrx
    yy = fvars ff `union` vv
    (xa,sa) = append xmax omax yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x0,s0) = buildb vv xc xc sc
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append xmax omax ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    t0 = [((jj, (bbv,ffv,ssv)), a1-b1) | jj <- vvll (topd (bmax `div` mmax) x0),
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,_,b1,_] = UV.toList ssv]
    t1 = [((jj, (bbv,ffv,ssv)), a1-b1) | jj <- vvll (topd (bmax `div` mmax) x1),
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,_,b1,_] = UV.toList ssv]
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    cross = parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvll = V.toList
    llvv = V.fromList
    qqvv = V.fromList . Set.toList

parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_ui :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_ui xmax omax bmax mmax uu vv ff hh hhx hhrr hhrrx
  | Set.size vv' < 2 = ([],0)
  | ff == fudEmpty = (res (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (res (topd (bmax `div` mmax) x1), s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    vv' = meff hhx vv
    (xc,sc) = cross xmax omax vv' hh hhx hhrr hhrrx
    yy = fvars ff `union` vv'
    (xa,sa) = append xmax omax yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x0,s0) = buildb vv' xc xc sc
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append xmax omax ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    res xx = [((jj, (bbv,ffv,ssv)), a1-b1) | jj <- vvll xx,
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,_,b1,_] = UV.toList ssv]
    meff hhx vv = Set.fromList [v | v <- qqll vv, let i = hhxv hhx Map.! v, 
      length (List.filter (/=0) (UV.toList (hhxi hhx V.! i))) > 1]
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    cross = parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    hhxv = histogramRepaRedsMapVarInt
    hhxi = histogramRepaRedsVectorArray
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    union = Set.union
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    sgl = Set.singleton
    vvll = V.toList
    qqvv = V.fromList . Set.toList

parametersSystemsBuilderDerivedVarsHighestRepa :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsHighestRepa wmax omax uu vv ff hh hhx hhrr hhrrx
  | wmax < 0 || omax < 0 = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | not (fvars ff `subset` uvars uu) = Nothing
  | otherwise = Just $ maxfst $ buildd (fvars ff `minus` vv) (init (fder ff)) []
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = [((sgl w, hempty, hempty),(0,0,0,0)) | w <- qqll vv]
    buildd ww qq nn = if mm /= [] then buildd ww mm (nn List.++ mm) else (final nn) 
      where
        pp = llqq [jj | ((kk,_,_),_) <- qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ [((jj, bb, bbrr), ((a-b)/c,-l,-b/c,-u)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax, fder (depends ff jj) == jj,
          let l =  sumlayer ff jj, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    final = List.filter (\((kk,_,_),_) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    depends = fudsVarsDepends
    fder = fudsDerived
    fvars = fudsVars
    vars = histogramsVars
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    maxfst mm = List.map (\((a,_,_,_),x) -> (x,a)) $ take 1 $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))
    sumlayer ff kk = sum [layer ff (sgl w) | w <- qqll kk]
    layer = fudsSetVarsLayer
    add xx x = x `Set.insert` xx
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    empty = Set.empty
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa wmax omax uu vv ff hh hhx hhrr hhrrx
  | wmax < 0 || omax < 0 = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | not (fvars ff `subset` uvars uu) = Nothing
  | otherwise = Just $ buildffdervar wmax omax uu vv ff hh hhx hhrr hhrrx
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    buildffdervar = parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_u
    fvars = fudsVars
    uvars = systemsVars
    subset = Set.isSubsetOf
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_u :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_u wmax omax uu vv ff hh hhx hhrr hhrrx = 
    fst $ buildffdervar wmax omax uu vv ff hh hhx hhrr hhrrx   
  where
    buildffdervar = parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_ui

parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_ui :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, HistogramRepa, HistogramRepa), Double)],Integer)
parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_ui wmax omax uu vv ff hh hhx hhrr hhrrx
  | yy == Set.empty = ([],0)
  | otherwise = (res (maxd x1),s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    cc = Set.fromList [(w,u) | w <- Set.toList (fvars ff `minus` vv), let gg = depends ff w, 
                               u <- Set.toList (fvars gg `minus` vv), u /= w]
    yy = fvars ff `minus` vv
    (xa,sa) = append wmax omax cc yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append wmax omax cc ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    res xx = [((jj, bb, bbrr), (a-b)/c) | jj <- vvll xx, let u = vol uu jj, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepaStorablesReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedExcludeHiddenDenseTop_u
    depends ff w = fudsVarsDepends ff (Set.singleton w)
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    maxd mm = snd $ V.unzip $ vectorPairsTop 1 mm
    minus = Set.difference
    sgl = Set.singleton
    vvll = V.toList
    qqvv = V.fromList . Set.toList



parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa wmax omax uu vv ff hh hhx hhrr hhrrx
  | wmax < 0 || omax < 0 = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | not (fvars ff `subset` uvars uu) = Nothing
  | otherwise = Just $ buildffdervar wmax omax uu vv ff hh hhx hhrr hhrrx
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    buildffdervar = parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_u
    fvars = fudsVars
    uvars = systemsVars
    subset = Set.isSubsetOf
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_1 :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_1 wmax omax uu vv ff hh hhx hhrr hhrrx
  | wmax < 0 || omax < 0 = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | not (fvars ff `subset` uvars uu) = Nothing
  | otherwise = Just $ maxfst $ buildd (fvars ff `minus` vv) (init (fder ff)) []
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = [((sgl w, hempty, hempty),(0,0,0)) | w <- qqll vv]
    buildd ww qq nn = if mm /= [] then buildd ww mm (nn List.++ mm) else (final nn) 
      where
        pp = llqq [jj | ((kk,_,_),_) <- qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ [((jj, bb, bbrr), ((a-b)/c,-b/c,-u)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax,
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    final = List.filter (\((kk,_,_),_) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    fder = fudsDerived
    fvars = fudsVars
    vars = histogramsVars
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    maxfst mm = List.map (\((a,_,_),x) -> (x,a)) $ take 1 $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))
    add xx x = x `Set.insert` xx
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    empty = Set.empty
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_2 :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_2 wmax omax uu vv ff hh hhx hhrr hhrrx
  | wmax < 0 || omax < 0 = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | not (fvars ff `subset` uvars uu) = Nothing
  | otherwise = Just $ V.toList $ maxfst $ buildd (fvars ff `minus` vv) (init (fder ff)) V.empty
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = V.fromListN (card vv) [((0,0,0),(sgl w, hempty, hempty)) | w <- qqll vv]
    buildd ww qq nn = if (not (V.null mm)) then buildd ww mm (nn V.++ mm) else (final nn) 
      where
        pp = llqq [jj | (_,(kk,_,_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ V.fromListN (card pp) [(((a-b)/c,-b/c,-u),(jj, bb, bbrr)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax,
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    final = V.filter (\(_,(kk,_,_)) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    fder = fudsDerived
    fvars = fudsVars
    vars = histogramsVars
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = vectorPairsTop (fromInteger amax) mm
    maxfst mm = V.map (\((a,_,_),x) -> (x,a)) $ vectorPairsTop 1 mm
    add xx x = x `Set.insert` xx
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    empty = Set.empty
    vvqq = Set.fromList . V.toList

parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_u :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_u wmax omax uu vv ff hh hhx hhrr hhrrx = 
    fst $ buildffdervar wmax omax uu vv ff hh hhx hhrr hhrrx   
  where
    buildffdervar = parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui

parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_u_1 :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_u_1 wmax omax uu vv ff hh hhx hhrr hhrrx = 
  V.toList $ maxfst $ buildd (fvars ff `minus` vv) (init (fder ff)) V.empty
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = V.fromListN (card vv) [((0,0,0),(sgl w, hempty, hempty)) | w <- qqll vv]
    buildd ww qq nn = if (not (V.null mm)) then buildd ww mm (nn V.++ mm) else (final nn) 
      where
        pp = llqq [jj | (_,(kk,_,_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ V.fromListN (card pp) [(((a-b)/c,-b/c,-u),(jj, bb, bbrr)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax,
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    final = V.filter (\(_,(kk,_,_)) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    maxfst mm = V.map (\((a,_,_),x) -> (x,a)) $ vectorPairsTop 1 mm
    add xx x = x `Set.insert` xx
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, HistogramRepa, HistogramRepa), Double)],Integer)
parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui wmax omax uu vv ff hh hhx hhrr hhrrx
  | yy == Set.empty = ([],0)
  | otherwise = (res (maxd x1),s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    yy = fvars ff `minus` vv
    (xa,sa) = append wmax omax yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append wmax omax ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    res xx = [((jj, bb, bbrr), (a-b)/c) | jj <- vvll xx, let u = vol uu jj, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepaStorablesReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop_u
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    maxd mm = snd $ V.unzip $ vectorPairsTop 1 mm
    minus = Set.difference
    sgl = Set.singleton
    vvll = V.toList
    qqvv = V.fromList . Set.toList

parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui_1 :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, HistogramRepa, HistogramRepa), Double)],Integer)
parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui_1 wmax omax uu vv ff hh hhx hhrr hhrrx = 
    (V.toList (maxfst x1),s1)
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    (x1,s1) =  buildd (fvars ff `minus` vv) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),(sgl w, hempty, hempty)) | w <- qqll vv]
    buildd ww qq nn s2 = 
        if (not (V.null mm)) then buildd ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,(kk,_,_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [(((a-b)/c,-b/c,-u),(jj, bb, bbrr)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax,
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
        mm = top omax x2
    final = V.filter (\(_,(kk,_,_)) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce_3
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    maxfst mm = V.map (\((a,_,_),x) -> (x,a)) $ vectorPairsTop 1 mm
    add xx x = x `Set.insert` xx
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui_2 :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, HistogramRepa, HistogramRepa), Double)],Integer)
parametersSystemsBuilderDerivedVarsHighestNoSumlayerIncludeHiddenRepa_ui_2 wmax omax uu vv ff hh hhx hhrr hhrrx = 
    (V.toList (maxfst x1),s1)
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed aa)) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed aarr)) :: SV.Vector CLLong
    (x1,s1) =  buildd (fvars ff `minus` vv) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),(sgl w, hempty, hempty)) | w <- qqll vv]
    buildd ww qq nn s2 = 
        if (not (V.null mm)) then buildd ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,(kk,_,_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [(((a-b)/c,-b/c,-u),(jj, bb, bbrr)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax,
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
        mm = top omax x2
    final = V.filter (\(_,(kk,_,_)) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepaStorablesReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    maxfst mm = V.map (\((a,_,_),x) -> (x,a)) $ vectorPairsTop 1 mm
    add xx x = x `Set.insert` xx
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsPartitionerRepa :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  Maybe [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))]
parametersSystemsPartitionerRepa mmax umax pmax uu kk bb y1
  | umax < 0 || mmax < 0 || pmax < 0 = Nothing
  | not (vvqq vbb `subset` uvars uu && kk `subset` vvqq vbb) = Nothing
  | otherwise = Just $ parter mmax umax pmax uu kk bb y1
  where
    (rrv,_,_) = bb
    HistogramRepaVec vbb _ _ _ _ = rrv 
    parter = parametersSystemsPartitionerRepa_u 
    uvars = systemsVars
    subset = Set.isSubsetOf
    vvqq = Set.fromList . V.toList

parametersSystemsPartitionerRepa_1 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> (HistogramRepaVec, UV.Vector Double) -> Double ->
  Maybe [(Set.Set (Set.Set Variable),HistogramRepaVec)]
parametersSystemsPartitionerRepa_1 mmax umax pmax uu kk (rrv,av) y1
  | umax < 0 || mmax < 0 || pmax < 0 = Nothing
  | not (vvqq vbb `subset` uvars uu && kk `subset` vvqq vbb) = Nothing
  | otherwise = Just $ topd pmax [((yy, ccv), ((y1-a2+b2)/c, b2, -m)) |
        yy <- stirsll kk mmax, dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv, let [ccx, ccrrx] = V.toList (rrvvrr ccv),
        let a2 = sumfacln ccx, let b2 = sumfacln ccrrx, let c = v ** (1/m)]
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    v = fromIntegral $ vol uu kk
    ppxx pp rrv = histogramRepaRedVecsIndependent_u $ setSetVarsHistogramRepaVecsPartitionRedVec_u pp rrv
    sumfacln rr = UV.sum $ UV.map (\x -> logGamma (x + 1)) rr
    facln x = logGamma (x + 1)
    rrvvrr = histogramRepaVecsArray
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip mm
    flip = List.map (\(a,b) -> (b,a))
    subset = Set.isSubsetOf
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    vvqq = Set.fromList . V.toList

parametersSystemsPartitionerRepa_2 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> (HistogramRepaVec, UV.Vector Double) -> Double ->
  Maybe [(Set.Set (Set.Set Variable),HistogramRepaVec)]
parametersSystemsPartitionerRepa_2 mmax umax pmax uu kk (rrv,av) y1
  | umax < 0 || mmax < 0 || pmax < 0 = Nothing
  | not (vvqq vbb `subset` uvars uu && kk `subset` vvqq vbb) = Nothing
  | otherwise = Just $ mm'
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    v = fromIntegral $ vol uu kk
    mm = topd pmax [((yy, ccv), ((y1-a2+b2)/c, b2, -m)) |
        yy <- stirsll kk mmax, dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv, let [ccx, ccrrx] = V.toList (rrvvrr ccv),
        let a2 = sumfacln ccx, let b2 = sumfacln ccrrx, let c = v ** (1/m)]
    mm' = [(yy, HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]))  | 
            (yy,HistogramRepaVec vcc mcc z scc rcc) <- mm, 
            let [ccx, ccrrx] = V.toList rcc, let [cc, ccrr] = V.toList (rrvvrr (pprr yy nnv))]
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv = histogramRepaRedVecsIndependent_u $ setSetVarsHistogramRepaVecsPartitionRedVec_u pp rrv
    sumfacln rr = UV.sum $ UV.map (\x -> logGamma (x + 1)) rr
    facln x = logGamma (x + 1)
    rrvvrr = histogramRepaVecsArray
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip mm
    flip = List.map (\(a,b) -> (b,a))
    subset = Set.isSubsetOf
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    vvqq = Set.fromList . V.toList

parametersSystemsPartitionerRepa_3 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> (HistogramRepaVec, UV.Vector Double) -> Double ->
  Maybe [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))]
parametersSystemsPartitionerRepa_3 mmax umax pmax uu kk (rrv,av) y1
  | umax < 0 || mmax < 0 || pmax < 0 = Nothing
  | not (vvqq vbb `subset` uvars uu && kk `subset` vvqq vbb) = Nothing
  | otherwise = Just $ mm'
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    v = fromIntegral $ vol uu kk
    mm = topd pmax [((yy, (ccv,ffv)), ((y1-a2+b2)/c, b2, -m)) |
        yy <- stirsll kk mmax, dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv, let ffv = rrvffv ccv, 
        let [a2, b2] = UV.toList (rrvsum ffv), let c = v ** (1/m)]
    mm' = [(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vff mff 1 sff (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              (yy,(HistogramRepaVec vcc mcc z scc rcc, HistogramRepaVec vff mff _ sff rff)) <- mm, 
              let [ccx, ccrrx] = V.toList rcc, let [cc, ccrr] = V.toList (rrvvrr (pprr yy nnv)), 
              let [ffx, ffrrx] = V.toList rff, let ff = facln cc, let ffrr = facln ccrr]
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv = histogramRepaRedVecsIndependent_u $ setSetVarsHistogramRepaVecsPartitionRedVec_u pp rrv
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip mm
    flip = List.map (\(a,b) -> (b,a))
    subset = Set.isSubsetOf
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    vvqq = Set.fromList . V.toList

parametersSystemsPartitionerRepa_4 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  Maybe [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))]
parametersSystemsPartitionerRepa_4 mmax umax pmax uu kk (rrv,ggv,ssv) y1
  | umax < 0 || mmax < 0 || pmax < 0 = Nothing
  | not (vvqq vbb `subset` uvars uu && kk `subset` vvqq vbb) = Nothing
  | otherwise = Just $ mm3
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    [ra1,ra2,rb1,rb2] = UV.toList ssv 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    n = toInteger $ UV.length sbb
    n' = fromIntegral n
    v = fromIntegral $ vol uu kk
    inc = n <= mmax
    mmax' = if inc then n-1 else mmax :: Integer
    mm1 = [((yy, (ccv,ffv), False), ((y1-a2+b2)/c, b2, -m)) |
        yy <- stirsll kk mmax', dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv, let ffv = rrvffv ccv, 
        let [a2, b2] = UV.toList (rrvsum ffv), let c = v ** (1/m)]
    mm2 = topd pmax $ if inc then ((self kk, (rrv,rrv), True),((y1-ra2+rb2)/(v**(1/n')), rb2, -n')) : mm1 else mm1
    mm3 = ([(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vff mff 1 sff (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              (yy,(HistogramRepaVec vcc mcc z scc rcc, HistogramRepaVec vff mff _ sff rff), isself) <- mm2, not isself,
              let [ccx, ccrrx] = V.toList rcc, let [cc, ccrr] = V.toList (rrvvrr (pprr yy nnv)), 
              let [ffx, ffrrx] = V.toList rff, let ff = facln cc, let ffrr = facln ccrr] List.++ 
           [(yy, (rrv,ggv)) | (yy, _, isself) <- mm2, isself])
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv = histogramRepaRedVecsIndependent_u $ setSetVarsHistogramRepaVecsPartitionRedVec_u pp rrv
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip mm
    flip = List.map (\(a,b) -> (b,a))
    self = Set.map Set.singleton
    subset = Set.isSubsetOf
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    vvqq = Set.fromList . V.toList

parametersSystemsPartitionerRepa_5 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  Maybe [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))]
parametersSystemsPartitionerRepa_5 mmax umax pmax uu kk (rrv,ggv,ssv) y1
  | umax < 0 || mmax < 0 || pmax < 0 = Nothing
  | not (vvqq vbb `subset` uvars uu && kk `subset` vvqq vbb) = Nothing
  | otherwise = Just $ mm3
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    [ra1,ra2,rb1,rb2] = UV.toList ssv 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    n = toInteger $ UV.length sbb
    n' = fromIntegral n
    v = fromIntegral $ vol uu kk
    inc = n <= mmax
    mmax' = if inc then n-1 else mmax :: Integer
    mm1 = [(((y1-a2+b2)/c, b2, -m),(yy, (ccv,ffv), False)) |
        yy <- stirsll kk mmax', dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv, let ffv = rrvffv ccv, 
        let [a2, b2] = UV.toList (rrvsum ffv), let c = v ** (1/m)]
    mm2 = topd pmax $ if inc then (((y1-ra2+rb2)/(v**(1/n')), rb2, -n'),(self kk, (rrv,rrv), True)) : mm1 else mm1
    mm3 = ([(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vff mff 1 sff (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              (yy,(HistogramRepaVec vcc mcc z scc rcc, HistogramRepaVec vff mff _ sff rff), isself) <- mm2, not isself,
              let [ccx, ccrrx] = V.toList rcc, let [cc, ccrr] = V.toList (rrvvrr (pprr yy nnv)), 
              let [ffx, ffrrx] = V.toList rff, let ff = facln cc, let ffrr = facln ccrr] List.++ 
           [(yy, (rrv,ggv)) | (yy, _, isself) <- mm2, isself])
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv = histogramRepaRedVecsIndependent_u $ setSetVarsHistogramRepaVecsPartitionRedVec_u pp rrv
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax = V.toList . snd . V.unzip . vectorPairsTop (fromInteger amax) . V.fromList
    self = Set.map Set.singleton
    subset = Set.isSubsetOf
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    vvqq = Set.fromList . V.toList

parametersSystemsPartitionerRepa_u :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))]
parametersSystemsPartitionerRepa_u mmax umax pmax uu kk bb y1 = 
    fst $ parter mmax umax pmax uu kk bb y1
  where
    parter = parametersSystemsPartitionerRepa_ui

parametersSystemsPartitionerRepa_u_1 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))]
parametersSystemsPartitionerRepa_u_1 mmax umax pmax uu kk (rrv,ggv,ssv) y1 = mm3
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    [ra1,ra2,rb1,rb2] = UV.toList ssv 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    n = toInteger $ UV.length sbb
    n' = fromIntegral n
    v = fromIntegral $ vol uu kk
    inc = n <= mmax
    mmax' = if inc then n-1 else mmax :: Integer
    mm1 = [(((y1-a2+b2)/c, b2, -m),(yy, (ccv,ffv), False)) |
        yy <- stirsll kk mmax', dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv, let ffv = rrvffv ccv, 
        let [a2, b2] = UV.toList (rrvsum ffv), let c = v ** (1/m)]
    mm2 = topd pmax $ if inc then (((y1-ra2+rb2)/(v**(1/n')), rb2, -n'),(self kk, (rrv,rrv), True)) : mm1 else mm1
    mm3 = ([(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vff mff 1 sff (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              (yy,(HistogramRepaVec vcc mcc z scc rcc, HistogramRepaVec vff mff _ sff rff), isself) <- mm2, not isself,
              let [ccx, ccrrx] = V.toList rcc, let [cc, ccrr] = V.toList (rrvvrr (pprr yy nnv)), 
              let [ffx, ffrrx] = V.toList rff, let ff = facln cc, let ffrr = facln ccrr] List.++ 
           [(yy, (rrv,ggv)) | (yy, _, isself) <- mm2, isself])
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv = histogramRepaRedVecsIndependent_u $ setSetVarsHistogramRepaVecsPartitionRedVec_u pp rrv
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr
    vol uu vv = systemsSetVarsVolume_u uu vv
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax = V.toList . snd . V.unzip . vectorPairsTop (fromInteger amax) . V.fromList
    self = Set.map Set.singleton
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList

parametersSystemsPartitionerRepa_ui :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  ([(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))],Integer)
parametersSystemsPartitionerRepa_ui mmax umax pmax uu kk (rrv,_,_) y1 = (mm3, q)
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, _, bbrr, _] = V.toList rbb 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    vsbb = SV.unsafeCast (UV.convert bb) :: SV.Vector CDouble
    vsbbrr = SV.unsafeCast (UV.convert bbrr) :: SV.Vector CDouble
    n = toInteger $ UV.length sbb
    (mm2,q) = rrvqqy mmax umax pmax rrv y1
    mm3 = [(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vcc mcc 1 scc (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              yy <- Set.toList mm2, 
              let ccv = pprr yy nnv,
              let HistogramRepaVec vcc mcc _ scc rcc = ccv,
              let [cc, ccrr] = V.toList rcc, 
              let [ccx, ccrrx] = V.toList (rrvvrr (ppxx yy nnv vsbb vsbbrr)),
              let ff = facln cc, let ffrr = facln ccrr,
              let ffx = facln ccx, let ffrrx = facln ccrrx]
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv vsaa vsaarr = setSetVarsHistogramRepaPairStorablesPartitionIndependentPair_u pp rrv vsaa vsaarr
    rrvqqy = parametersHistogramRepaVecsSetTuplePartitionTop_u 
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr

parametersSystemsPartitionerRepa_ui_1 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  ([(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))],Integer)
parametersSystemsPartitionerRepa_ui_1 mmax umax pmax uu kk (rrv,ggv,ssv) y1 = (mm3, toInteger (length mm1))
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    [ra1,ra2,rb1,rb2] = UV.toList ssv 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    n = toInteger $ UV.length sbb
    n' = fromIntegral n
    v = fromIntegral $ vol uu kk
    inc = n <= mmax
    mmax' = if inc then n-1 else mmax :: Integer
    mm1 = [(((y1-a2+b2)/c, b2, -m),(yy, (ccv,ffv), False)) |
        yy <- stirsll kk mmax', dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv, let ffv = rrvffv ccv, 
        let [a2, b2] = UV.toList (rrvsum ffv), let c = v ** (1/m)]
    mm2 = topd pmax $ if inc then (((y1-ra2+rb2)/(v**(1/n')), rb2, -n'),(self kk, (rrv,rrv), True)) : mm1 else mm1
    mm3 = ([(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vff mff 1 sff (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              (yy,(HistogramRepaVec vcc mcc z scc rcc, HistogramRepaVec vff mff _ sff rff), isself) <- mm2, not isself,
              let [ccx, ccrrx] = V.toList rcc, let [cc, ccrr] = V.toList (rrvvrr (pprr yy nnv)), 
              let [ffx, ffrrx] = V.toList rff, let ff = facln cc, let ffrr = facln ccrr] List.++ 
           [(yy, (rrv,ggv)) | (yy, _, isself) <- mm2, isself])
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv = histogramRepaRedVecsIndependent_u $ setSetVarsHistogramRepaVecsPartitionRedVec_u pp rrv
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr
    vol uu vv = systemsSetVarsVolume_u uu vv
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax = V.toList . snd . V.unzip . vectorPairsTop (fromInteger amax) . V.fromList
    self = Set.map Set.singleton
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList

parametersSystemsPartitionerRepa_ui_2 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  ([(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))],Integer)
parametersSystemsPartitionerRepa_ui_2 mmax umax pmax uu kk (rrv,ggv,ssv) y1 = (mm3, toInteger (length mm1))
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    [ra1,ra2,rb1,rb2] = UV.toList ssv 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    n = toInteger $ UV.length sbb
    n' = fromIntegral n
    v = fromIntegral $ vol uu kk
    inc = n <= mmax
    mmax' = if inc then n-1 else mmax :: Integer
    mm1 = [(((y1-a2+b2)/c, b2, -m),(yy, (ccv,ffv), False)) |
        yy <- stirsll kk mmax', dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv, let ffv = rrvffv ccv, 
        let [a2, b2] = UV.toList (rrvsum ffv), let c = v ** (1/m)]
    mm2 = topd pmax $ if inc then (((y1-ra2+rb2)/(v**(1/n')), rb2, -n'),(self kk, (rrv,rrv), True)) : mm1 else mm1
    mm3 = ([(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vff mff 1 sff (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              (yy,(HistogramRepaVec vcc mcc z scc rcc, HistogramRepaVec vff mff _ sff rff), isself) <- mm2, not isself,
              let [ccx, ccrrx] = V.toList rcc, let [cc, ccrr] = V.toList (rrvvrr (pprr yy nnv)), 
              let [ffx, ffrrx] = V.toList rff, let ff = facln cc, let ffrr = facln ccrr] List.++ 
           [(yy, (rrv,ggv)) | (yy, _, isself) <- mm2, isself])
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv = setSetVarsHistogramRepaVecsPartitionIndependentVec_u pp rrv
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr
    vol uu vv = systemsSetVarsVolume_u uu vv
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax = V.toList . snd . V.unzip . vectorPairsTop (fromInteger amax) . V.fromList
    self = Set.map Set.singleton
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList

parametersSystemsPartitionerRepa_ui_3 :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  ([(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))],Integer)
parametersSystemsPartitionerRepa_ui_3 mmax umax pmax uu kk (rrv,ggv,ssv) y1 = (mm3, toInteger (length mm1))
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, bbx, bbrr, bbrrx] = V.toList rbb 
    [ra1,ra2,rb1,rb2] = UV.toList ssv 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    vsbb = SV.unsafeCast (UV.convert bb) :: SV.Vector CDouble
    vsbbrr = SV.unsafeCast (UV.convert bbrr) :: SV.Vector CDouble
    n = toInteger $ UV.length sbb
    n' = fromIntegral n
    v = fromIntegral $ vol uu kk
    inc = n <= mmax
    mmax' = if inc then n-1 else mmax :: Integer
    mm1 = [(((y1-a2+b2)/c, b2, -m),(yy, (ccv,ffv), False)) |
        yy <- stirsll kk mmax', dim yy >= 2, and [vol uu jj <= umax | jj <- qqll yy],
        let m = fromIntegral $ dim yy,
        let ccv = ppxx yy nnv vsbb vsbbrr, let ffv = rrvffv ccv, 
        let [a2, b2] = UV.toList (rrvsum ffv), let c = v ** (1/m)]
    mm2 = topd pmax $ if inc then (((y1-ra2+rb2)/(v**(1/n')), rb2, -n'),(self kk, (rrv,rrv), True)) : mm1 else mm1
    mm3 = ([(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vff mff 1 sff (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              (yy,(HistogramRepaVec vcc mcc z scc rcc, HistogramRepaVec vff mff _ sff rff), isself) <- mm2, not isself,
              let [ccx, ccrrx] = V.toList rcc, let [cc, ccrr] = V.toList (rrvvrr (pprr yy nnv)), 
              let [ffx, ffrrx] = V.toList rff, let ff = facln cc, let ffrr = facln ccrr] List.++ 
           [(yy, (rrv,ggv)) | (yy, _, isself) <- mm2, isself])
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv vsaa vsaarr = setSetVarsHistogramRepaPairStorablesPartitionIndependentPair_u pp rrv vsaa vsaarr
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr
    vol uu vv = systemsSetVarsVolume_u uu vv
    stirsll vv bmax = Set.toList $ setsSetPartitionLimited vv bmax
    dim = toInteger . Set.size
    ssgl = stateSingleton
    sempty = stateEmpty
    topd amax = V.toList . snd . V.unzip . vectorPairsTop (fromInteger amax) . V.fromList
    self = Set.map Set.singleton
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList

parametersSystemsPartitionerMaxRollByMRepa :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  Maybe [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))]
parametersSystemsPartitionerMaxRollByMRepa mmax umax pmax uu kk bb y1
  | umax < 0 || mmax < 0 || pmax < 0 = Nothing
  | not (vvqq vbb `subset` uvars uu && kk `subset` vvqq vbb) = Nothing
  | otherwise = Just $ parter mmax umax pmax uu kk bb y1
  where
    (rrv,_,_) = bb
    HistogramRepaVec vbb _ _ _ _ = rrv 
    parter = parametersSystemsPartitionerMaxRollByMRepa_u 
    uvars = systemsVars
    subset = Set.isSubsetOf
    vvqq = Set.fromList . V.toList

parametersSystemsPartitionerMaxRollByMRepa_u :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))]
parametersSystemsPartitionerMaxRollByMRepa_u mmax umax pmax uu kk bb y1 = 
    fst $ parter mmax umax pmax uu kk bb y1
  where
    parter = parametersSystemsPartitionerMaxRollByMRepa_ui

parametersSystemsPartitionerMaxRollByMRepa_ui :: 
  Integer -> Integer -> Integer -> System -> Set.Set Variable -> 
  (HistogramRepaVec, HistogramRepaVec, UV.Vector Double) -> Double ->
  ([(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))],Integer)
parametersSystemsPartitionerMaxRollByMRepa_ui mmax umax pmax uu kk (rrv,_,_) y1 = (mm3, q)
  where
    HistogramRepaVec vbb mbb z sbb rbb = rrv 
    [bb, _, bbrr, _] = V.toList rbb 
    nnv = HistogramRepaVec vbb mbb z sbb (V.fromListN 2 [bb, bbrr])
    vsbb = SV.unsafeCast (UV.convert bb) :: SV.Vector CDouble
    vsbbrr = SV.unsafeCast (UV.convert bbrr) :: SV.Vector CDouble
    n = toInteger $ UV.length sbb
    (mm2,q) = rrvqqy mmax umax pmax rrv y1
    mm3 = [(yy, (HistogramRepaVec vcc mcc z scc (V.fromListN 4 [cc, ccx, ccrr, ccrrx]), 
            HistogramRepaVec vcc mcc 1 scc (V.fromListN 4 [ff, ffx, ffrr, ffrrx]))) | 
              yy <- Set.toList mm2, 
              let ccv = pprr yy nnv,
              let HistogramRepaVec vcc mcc _ scc rcc = ccv,
              let [cc, ccrr] = V.toList rcc, 
              let [ccx, ccrrx] = V.toList (rrvvrr (ppxx yy nnv vsbb vsbbrr)),
              let ff = facln cc, let ffrr = facln ccrr,
              let ffx = facln ccx, let ffrrx = facln ccrrx]
    pprr pp rrv = setSetVarsHistogramRepaVecsPartitionVec_u pp rrv
    ppxx pp rrv vsaa vsaarr = setSetVarsHistogramRepaPairStorablesPartitionIndependentPair_u pp rrv vsaa vsaarr
    rrvqqy = parametersHistogramRepaVecsSetTuplePartitionTopByM_u 
    rrvvrr = histogramRepaVecsArray
    facln rr = UV.map (\x -> logGamma (x + 1)) rr



parametersRollerRepa :: 
  Integer -> [(Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec))] -> Maybe [(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))]
parametersRollerRepa pmax qq
  | pmax < 0 = Nothing
  | otherwise = Just $ fst $ unzip $ topd pmax (rollb mm mm)
  where
    mm = [(((yy,pp),((ccv,ffv),xxv,(a,w,m))),a/c) | (yy,(ccv,ffv)) <- qq,
          let xxv = reds ffv, let a = UV.sum (vec (sing 0 xxv)),
          let scc = sh ccv, let w = R.size scc, let m = R.rank scc,
          let c = fromIntegral w ** (1 / fromIntegral m), 
          let pp = V.map (\d -> UV.enumFromN 0 d) (UV.convert scc)]
    rollb qq pp
      | mm /= [] = rollb mm' (pp List.++ mm')
      | otherwise = pp         
      where
        mm = top pmax $ [(((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),a'/c') | 
               (((yy,pp),((ccv,ffv),xxv,(a,w,m))),_) <- qq, let scc = sh ccv, 
               v <- [0..m-1], let d = scc UV.! v, d > 2, 
               let w' = w * (d-1) `div` d, let c' = fromIntegral w' ** (1 / fromIntegral m), 
               let (rrv,(rs,rt)) = rollv v ccv, let ggv = faclns rrv, 
               let av = sumv a (sing v xxv) (rs,rt) (sing4 v ggv),
               (q,(s,t,a')) <- zip [0..] (UV.toList (UV.zip3 rs rt av))]
        mm' =  [(((yy,pp'),((ccv',ffv'),xxv',(a',w',m))),b') | 
                 (((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),b') <- mm,
                 let ccv' = copyv v s t q ccv rrv, let ffv' = copyv v s t q ffv ggv, let xxv' = reds ffv',
                 let pp' = pp V.// [(v, rollr s t (pp V.! v))]]
    sh = histogramRepaVecsShape
    vec = R.toUnboxed . histogramRepasArray
    faclns = histogramRepaVecsFaclnsRepaVecs
    reds = histogramRepa4VecsRed_u
    sing = varsHistogramRepaRedsSingle_u
    sing4 = varsHistogramRepa4VecsReduceSingle_u
    rollv = varsHistogramRepaVecsRollVec_u
    sumv = sumsHistogramRepasRollMapPairsHistogramRepasSum_u
    copyv = varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u
    rollr s t = UV.map (\r -> if r > s then r-1 else (if r == s then t else r))
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))

parametersRollerMaximumRollRepa :: 
  (Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec)) -> 
  [(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))]
parametersRollerMaximumRollRepa qq = fst $ unzip $ topd (rollb mm mm)
  where
    mm = [(((yy,pp),((ccv,ffv),xxv,(a,w,m))),a/c) | let (yy,(ccv,ffv)) = qq, 
          let xxv = reds ffv, let a = UV.sum (vec (sing 0 xxv)),
          let scc = sh ccv, let w = R.size scc, let m = R.rank scc,
          let c = fromIntegral w ** (1 / fromIntegral m), 
          let pp = V.map (\d -> UV.enumFromN 0 d) (UV.convert scc)]
    rollb qq pp
      | mm /= [] = rollb mm' (pp List.++ mm')
      | otherwise = pp         
      where
        mm = top $ [(((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),a'/c') | 
               (((yy,pp),((ccv,ffv),xxv,(a,w,m))),_) <- qq, let scc = sh ccv, 
               v <- [0..m-1], let d = scc UV.! v, d > 2, 
               let w' = w * (d-1) `div` d, let c' = fromIntegral w' ** (1 / fromIntegral m), 
               let (rrv,(rs,rt)) = rollv v ccv, let ggv = faclns rrv, 
               let av = sumv a (sing v xxv) (rs,rt) (sing4 v ggv),
               (q,(s,t,a')) <- zip [0..] (UV.toList (UV.zip3 rs rt av))]
        mm' =  [(((yy,pp'),((ccv',ffv'),xxv',(a',w',m))),b') | 
                 (((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),b') <- mm,
                 let ccv' = copyv v s t q ccv rrv, let ffv' = copyv v s t q ffv ggv, let xxv' = reds ffv',
                 let pp' = pp V.// [(v, rollr s t (pp V.! v))]]
    sh = histogramRepaVecsShape
    vec = R.toUnboxed . histogramRepasArray
    faclns = histogramRepaVecsFaclnsRepaVecs
    reds = histogramRepa4VecsRed_u
    sing = varsHistogramRepaRedsSingle_u
    sing4 = varsHistogramRepa4VecsReduceSingle_u
    rollv = varsHistogramRepaVecsRollVec_u
    sumv = sumsHistogramRepasRollMapPairsHistogramRepasSum_u
    copyv = varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u
    rollr s t = UV.map (\r -> if r > s then r-1 else (if r == s then t else r))
    topd mm = snd $ unzip $ take 1 $ reverse $ sort $ flip $ mm
    top mm = flip $ take 1 $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))

parametersRollerMaximumRollExcludedSelfRepa :: 
  (Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec)) -> 
  [(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))]
parametersRollerMaximumRollExcludedSelfRepa qq = 
    fst $ parter qq
  where
    parter = parametersRollerMaximumRollExcludedSelfRepa_i

parametersRollerMaximumRollExcludedSelfRepa_1 :: 
  (Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec)) -> 
  [(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))]
parametersRollerMaximumRollExcludedSelfRepa_1 qq = fst $ unzip $ topd (rollb mm [])
  where
    mm = [(((yy,pp),((ccv,ffv),xxv,(a,w,m))),a/c) | let (yy,(ccv,ffv)) = qq, 
          let xxv = reds ffv, let a = UV.sum (vec (sing 0 xxv)),
          let scc = sh ccv, let w = R.size scc, let m = R.rank scc,
          let c = fromIntegral w ** (1 / fromIntegral m), 
          let pp = V.map (\d -> UV.enumFromN 0 d) (UV.convert scc)]
    rollb qq pp
      | mm /= [] = rollb mm' (pp List.++ mm')
      | otherwise = pp         
      where
        mm = top $ [(((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),a'/c') | 
               (((yy,pp),((ccv,ffv),xxv,(a,w,m))),_) <- qq, let scc = sh ccv, 
               v <- [0..m-1], let d = scc UV.! v, d > 2, 
               let w' = w * (d-1) `div` d, let c' = fromIntegral w' ** (1 / fromIntegral m), 
               let (rrv,(rs,rt)) = rollv v ccv, let ggv = faclns rrv, 
               let av = sumv a (sing v xxv) (rs,rt) (sing4 v ggv),
               (q,(s,t,a')) <- zip [0..] (UV.toList (UV.zip3 rs rt av))]
        mm' =  [(((yy,pp'),((ccv',ffv'),xxv',(a',w',m))),b') | 
                 (((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),b') <- mm,
                 let ccv' = copyv v s t q ccv rrv, let ffv' = copyv v s t q ffv ggv, let xxv' = reds ffv',
                 let pp' = pp V.// [(v, rollr s t (pp V.! v))]]
    sh = histogramRepaVecsShape
    vec = R.toUnboxed . histogramRepasArray
    faclns = histogramRepaVecsFaclnsRepaVecs
    reds = histogramRepa4VecsRed_u
    sing = varsHistogramRepaRedsSingle_u
    sing4 = varsHistogramRepa4VecsReduceSingle_u
    rollv = varsHistogramRepaVecsRollVec_u
    sumv = sumsHistogramRepasRollMapPairsHistogramRepasSum_u
    copyv = varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u
    rollr s t = UV.map (\r -> if r > s then r-1 else (if r == s then t else r))
    topd mm = snd $ unzip $ take 1 $ reverse $ sort $ flip $ mm
    top mm = flip $ take 1 $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))

parametersRollerMaximumRollExcludedSelfRepa_2 :: 
  (Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec)) -> 
  [(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))]
parametersRollerMaximumRollExcludedSelfRepa_2 qq = fst $ unzip $ topd (rollb mm [])
  where
    mm = [(((yy,pp),((ccv,ffv),xxv,(a,w,m))),a/c) | let (yy,(ccv,ffv)) = qq, 
          let xxv = reds ffv, let a = UV.sum (vec (sing 0 xxv)),
          let scc = sh ccv, let w = R.size scc, let m = R.rank scc,
          let c = fromIntegral w ** (1 / fromIntegral m), 
          let pp = V.map (\d -> UV.enumFromN 0 d) (UV.convert scc)]
    rollb qq pp
      | mm /= [] = rollb mm' (pp List.++ mm')
      | otherwise = pp         
      where
        mm = top $ [(((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),a'/c') | 
               (((yy,pp),((ccv,ffv),xxv,(a,w,m))),_) <- qq, let scc = sh ccv, 
               v <- [0..m-1], let d = scc UV.! v, d > 2, 
               let w' = w * (d-1) `div` d, let c' = fromIntegral w' ** (1 / fromIntegral m), 
               let (rrv,(rs,rt)) = rollv v ccv, let ggv = faclns rrv, 
               let av = sumv a (sing v xxv) (rs,rt) (sing4 v ggv),
               (q,(s,t,a')) <- zip [0..] (UV.toList (UV.zip3 rs rt av))]
        mm' =  [(((yy,pp'),((ccv',ffv'),xxv',(a',w',m))),b') | 
                 (((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),b') <- mm,
                 let ccv' = copyv v s t q ccv rrv, let ffv' = copyv v s t q ffv ggv, let xxv' = reds ffv',
                 let pp' = pp V.// [(v, rollr s t (pp V.! v))]]
    sh = histogramRepaVecsShape
    vec = R.toUnboxed . histogramRepasArray
    faclns = histogramRepaVecsFaclnsRepaVecs
    reds = histogramRepa4VecsRed_u
    sing = varsHistogramRepaRedsSingle_u
    sing4 = varsHistogramRepa4VecsReduceSingle_u
    rollv = varsHistogramRepaVecsRollVec_u
    sumv = sumsHistogramRepasRollMapPairsHistogramRepasSum_u
    copyv = varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u
    rollr s t = UV.map (\r -> if r > s then r-1 else (if r == s then t else r))
    topd mm = fst $ unzip $ top mm
    top mm 
      | l <= 0 = []
      | otherwise  = [V.unsafeIndex vv m]
      where
        l = length mm
        vv = V.fromListN l mm
        vv1 = V.imap (\i (a,b) -> (b,i)) vv     
        (_,m) = V.maximum vv1

parametersRollerMaximumRollExcludedSelfRepa_3 :: 
  (Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec)) -> 
  [(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))]
parametersRollerMaximumRollExcludedSelfRepa_3 qq = topd (rollb pp ccv ffv xxv a w [])
  where
    (!yy,(!ccv,!ffv)) = qq
    !xxv = reds ffv
    !a = UV.sum (vec (sing 0 xxv))
    !scc = sh ccv
    !w = R.size scc
    !m = R.rank scc
    !pp = V.map (\d -> UV.enumFromN 0 d) (UV.convert scc)
    rollb pp ccv ffv xxv a w nn
      | mm /= [] = rollb pp' ccv' ffv' xxv' a' w' (((yy,pp'),b'):nn)
      | otherwise = nn         
      where
        !scc = sh ccv
        !mm = top $ [((a',w',(v,s,t,q,rrv,ggv)),a'/c') | 
               v <- [0..m-1], let d = scc UV.! v, d > 2, 
               let w' = w * (d-1) `div` d, let c' = fromIntegral w' ** (1 / fromIntegral m), 
               let (rrv,(rs,rt)) = rollv v ccv, let ggv = faclns rrv, 
               let av = sumv a (sing v xxv) (rs,rt) (sing4 v ggv),
               (q,(s,t,a')) <- zip [0..] (UV.toList (UV.zip3 rs rt av))]
        ((a',w',(v,s,t,q,rrv,ggv)),b') = head mm
        ccv' = copyv v s t q ccv rrv
        ffv' = copyv v s t q ffv ggv
        xxv' = reds ffv'
        pp' = pp V.// [(v, rollr s t (pp V.! v))]
    sh = histogramRepaVecsShape
    vec = R.toUnboxed . histogramRepasArray
    faclns = histogramRepaVecsFaclnsRepaVecs
    reds = histogramRepa4VecsRed_u
    sing = varsHistogramRepaRedsSingle_u
    sing4 = varsHistogramRepa4VecsReduceSingle_u
    rollv = varsHistogramRepaVecsRollVec_u
    sumv = sumsHistogramRepasRollMapPairsHistogramRepasSum_u
    copyv = varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u
    rollr s t = UV.map (\r -> if r > s then r-1 else (if r == s then t else r))
    topd mm = fst $ unzip $ top mm
    top mm 
      | l <= 0 = []
      | otherwise  = [V.unsafeIndex vv m]
      where
        l = length mm
        vv = V.fromListN l mm
        vv1 = V.imap (\i (a,b) -> (b,i)) vv     
        (_,m) = V.maximum vv1

parametersRollerMaximumRollExcludedSelfRepa_i_2 :: 
  (Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec)) -> 
  ([(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))],Integer)
parametersRollerMaximumRollExcludedSelfRepa_i_2 qq = (topd z1,y1)
  where
    (!yy,(!ccv,!ffv)) = qq
    !xxv = reds ffv
    !a = UV.sum (vec (sing 0 xxv))
    !scc = sh ccv
    !w = R.size scc
    !m = R.rank scc
    !pp = V.map (\d -> UV.enumFromN 0 d) (UV.convert scc)
    (!z1,!y1) = rollb pp ccv ffv xxv a w [] 0
    rollb pp ccv ffv xxv a w nn y
      | mm /= [] = rollb pp' ccv' ffv' xxv' a' w' (((yy,pp'),b'):nn) y'
      | otherwise = (nn,y)         
      where
        !scc = sh ccv
        !mm = [((a',w',(v,s,t,q,rrv,ggv)),a'/c') | 
               v <- [0..m-1], let d = scc UV.! v, d > 2, 
               let w' = w * (d-1) `div` d, let c' = fromIntegral w' ** (1 / fromIntegral m), 
               let (rrv,(rs,rt)) = rollv v ccv, let ggv = faclns rrv, 
               let av = sumv a (sing v xxv) (rs,rt) (sing4 v ggv),
               (q,(s,t,a')) <- zip [0..] (UV.toList (UV.zip3 rs rt av))]
        ((a',w',(v,s,t,q,rrv,ggv)),b') = head $ top mm
        ccv' = copyv v s t q ccv rrv
        ffv' = copyv v s t q ffv ggv
        xxv' = reds ffv'
        pp' = pp V.// [(v, rollr s t (pp V.! v))]
        !y' = y + toInteger (length mm)
    sh = histogramRepaVecsShape
    vec = R.toUnboxed . histogramRepasArray
    faclns = histogramRepaVecsFaclnsRepaVecs
    reds = histogramRepa4VecsRed_u
    sing = varsHistogramRepaRedsSingle_u
    sing4 = varsHistogramRepa4VecsReduceSingle_u
    rollv = varsHistogramRepaVecsRollVec_u
    sumv = sumsHistogramRepasRollMapPairsHistogramRepasSum_u
    copyv = varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u
    rollr s t = UV.map (\r -> if r > s then r-1 else (if r == s then t else r))
    topd mm = fst $ unzip $ top mm
    top mm 
      | l <= 0 = []
      | otherwise  = [V.unsafeIndex vv m]
      where
        l = length mm
        vv = V.fromListN l mm
        vv1 = V.imap (\i (a,b) -> (b,i)) vv     
        (_,m) = V.maximum vv1

parametersRollerMaximumRollExcludedSelfRepa_i_1 :: 
  (Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec)) -> 
  ([(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))],Integer)
parametersRollerMaximumRollExcludedSelfRepa_i_1 qq = (fst (unzip (topd z1)),y1)
  where
    mm = [(((yy,pp),((ccv,ffv),xxv,(a,w,m))),a/c) | let (yy,(ccv,ffv)) = qq, 
          let xxv = reds ffv, let a = UV.sum (vec (sing 0 xxv)),
          let scc = sh ccv, let w = R.size scc, let m = R.rank scc,
          let c = fromIntegral w ** (1 / fromIntegral m), 
          let pp = V.map (\d -> UV.enumFromN 0 d) (UV.convert scc)]
    (z1,y1) = rollb mm [] 0
    rollb qq pp y
      | mm /= [] = rollb mm' (pp List.++ mm') (y + toInteger (length mm))
      | otherwise = (pp,y)
      where
        mm = [(((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),a'/c') | 
               (((yy,pp),((ccv,ffv),xxv,(a,w,m))),_) <- qq, let scc = sh ccv, 
               v <- [0..m-1], let d = scc UV.! v, d > 2, 
               let w' = w * (d-1) `div` d, let c' = fromIntegral w' ** (1 / fromIntegral m), 
               let (rrv,(rs,rt)) = rollv v ccv, let ggv = faclns rrv, 
               let av = sumv a (sing v xxv) (rs,rt) (sing4 v ggv),
               (q,(s,t,a')) <- zip [0..] (UV.toList (UV.zip3 rs rt av))]
        mm' =  [(((yy,pp'),((ccv',ffv'),xxv',(a',w',m))),b') | 
                 (((yy,pp),((ccv,ffv),(a',w',m),(v,s,t,q,rrv,ggv))),b') <- top mm,
                 let ccv' = copyv v s t q ccv rrv, let ffv' = copyv v s t q ffv ggv, let xxv' = reds ffv',
                 let pp' = pp V.// [(v, rollr s t (pp V.! v))]]
    sh = histogramRepaVecsShape
    vec = R.toUnboxed . histogramRepasArray
    faclns = histogramRepaVecsFaclnsRepaVecs
    reds = histogramRepa4VecsRed_u
    sing = varsHistogramRepaRedsSingle_u
    sing4 = varsHistogramRepa4VecsReduceSingle_u
    rollv = varsHistogramRepaVecsRollVec_u
    sumv = sumsHistogramRepasRollMapPairsHistogramRepasSum_u
    copyv = varsSourcesTargetsRollsHistogramRepaVecsHistogramRepaVecRollsCopyVec_u
    rollr s t = UV.map (\r -> if r > s then r-1 else (if r == s then t else r))
    topd mm = fst $ unzip $ top mm
    top mm 
      | l <= 0 = []
      | otherwise  = [V.unsafeIndex vv m]
      where
        l = length mm
        vv = V.fromListN l mm
        vv1 = V.imap (\i (a,b) -> (b,i)) vv     
        (_,m) = V.maximum vv1

parametersRollerMaximumRollExcludedSelfRepa_i :: 
  (Set.Set (Set.Set Variable),(HistogramRepaVec,HistogramRepaVec)) -> 
  ([(Set.Set (Set.Set Variable),V.Vector (UV.Vector Int))],Integer)
parametersRollerMaximumRollExcludedSelfRepa_i (!yy,(!rrv,_)) = (ll,q)
  where
    (!tt,!q) = histogramRepaVecsRollMax rrv
    ll = if V.or (V.map (\vv -> UV.maximum vv < UV.length vv - 1) tt) then [(yy,tt)] else []

parametersSystemsLayererHighestRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  Maybe (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererHighestRepa wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 = Nothing
  | not (vvqq (hhvvr xx) `subset` uvars uu 
      && hhvvr xx == hhvvr xxrr && hhvvr xx == apvvr xxp && hhvvr xx == apvvr xxrrp 
      && vv `subset` vvqq (hhvvr xx)) = Nothing
  | otherwise = Just $ layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then 
        layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) else (uu,ff,mm) 
      where
        ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | 
               ((kk,bb),y1) <- buildfftup uu vv ff xx xxp xxrr xxrrp, 
               (yy,pp) <- roller (parter uu kk bb y1), 
               (jj,p) <- zip (qqll yy) (V.toList pp), let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        hh = qqff $ llqq $ fst $ unzip ll'
        uu' = uu `uunion` (lluu $ snd $ unzip ll')
        ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        xx' = apply xx ffr
        xxp' = historyRepasRed xx'
        xxrr' = apply xxrr ffr
        xxrrp' = historyRepasRed xxrr'
        gg = ff `funion` hh
        mm' = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      fromJust $ parametersSystemsBuilderTupleRepa xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = fromJust $ parametersSystemsPartitionerRepa mmax umax pmax uu kk bb y1
    roller qq = fromJust $ parametersRollerRepa pmax qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (List.map (\((kk,_,_),a) -> (kk,a)) $ 
      fromJust $ parametersSystemsBuilderDerivedVarsHighestRepa wmax omax uu vv ff xx xxp xxrr xxrrp)
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = fromJust $ systemsTransformsTransformRepa uu tt
    hhvvr = historyRepasVectorVar
    apvvr = histogramRepaRedsVectorVar
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    ttpp = transformsPartition
    und = transformsUnderlying
    trans = histogramsSetVarsTransform_u
    unit qq = listsHistogram_u $ List.map (\ss -> (ss,1)) $ qq
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    uvars = systemsVars
    uunion = pairSystemsUnion
    lluu = listsSystem_u
    nnww = ValInt . toInteger
    maxr mm = if mm /= [] then (last $ sort $ snd $ unzip $ mm) else 0
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    subset = Set.isSubsetOf
    sgl = Set.singleton
    qqll = Set.toList
    vvqq = Set.fromList . V.toList

parametersSystemsLayererMaximumRollHighestRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  Maybe (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollHighestRepa wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 = Nothing
  | not (vvqq (hhvvr xx) `subset` uvars uu 
      && hhvvr xx == hhvvr xxrr && hhvvr xx == apvvr xxp && hhvvr xx == apvvr xxrrp 
      && vv `subset` vvqq (hhvvr xx)) = Nothing
  | otherwise = Just $ layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then 
        layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) else (uu,ff,mm) 
      where
        ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | 
               ((kk,bb),y1) <- buildfftup uu vv ff xx xxp xxrr xxrrp, 
               qq <- parter uu kk bb y1, (yy,pp) <- roller qq, 
               (jj,p) <- zip (qqll yy) (V.toList pp), let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        hh = qqff $ llqq $ fst $ unzip ll'
        uu' = uu `uunion` (lluu $ snd $ unzip ll')
        ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        xx' = apply xx ffr
        xxp' = historyRepasRed xx'
        xxrr' = apply xxrr ffr
        xxrrp' = historyRepasRed xxrr'
        gg = ff `funion` hh
        mm' = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      fromJust $ parametersSystemsBuilderTupleRepa xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = fromJust $ parametersSystemsPartitionerRepa mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollRepa qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (List.map (\((kk,_,_),a) -> (kk,a)) $ 
      fromJust $ parametersSystemsBuilderDerivedVarsHighestRepa wmax omax uu vv ff xx xxp xxrr xxrrp)
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = fromJust $ systemsTransformsTransformRepa uu tt
    hhvvr = historyRepasVectorVar
    apvvr = histogramRepaRedsVectorVar
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    ttpp = transformsPartition
    und = transformsUnderlying
    trans = histogramsSetVarsTransform_u
    unit qq = listsHistogram_u $ List.map (\ss -> (ss,1)) $ qq
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    uvars = systemsVars
    uunion = pairSystemsUnion
    lluu = listsSystem_u
    nnww = ValInt . toInteger
    maxr mm = if mm /= [] then (last $ sort $ snd $ unzip $ mm) else 0
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    subset = Set.isSubsetOf
    sgl = Set.singleton
    qqll = Set.toList
    vvqq = Set.fromList . V.toList

parametersSystemsLayererMaximumRollExcludedSelfHighestRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  Maybe (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestRepa 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 = Nothing
  | not (vvqq (hhvvr xx) `subset` uvars uu 
      && hhvvr xx == hhvvr xxrr && hhvvr xx == apvvr xxp && hhvvr xx == apvvr xxrrp 
      && vv `subset` vvqq (hhvvr xx)) = Nothing
  | otherwise = Just $ parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
      wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
  where
    hhvvr = historyRepasVectorVar
    apvvr = histogramRepaRedsVectorVar
    uvars = systemsVars
    vvqq = Set.fromList . V.toList
    subset = Set.isSubsetOf

parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  Maybe (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_1 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 = Nothing
  | not (vvqq (hhvvr xx) `subset` uvars uu 
      && hhvvr xx == hhvvr xxrr && hhvvr xx == apvvr xxp && hhvvr xx == apvvr xxrrp 
      && vv `subset` vvqq (hhvvr xx)) = Nothing
  | otherwise = Just $ layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then 
        layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) else (uu,ff,mm) 
      where
        ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | 
               ((kk,bb),y1) <- buildfftup uu vv ff xx xxp xxrr xxrrp, 
               qq <- parter uu kk bb y1, (yy,pp) <- roller qq, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        hh = qqff $ llqq $ fst $ unzip ll'
        uu' = uu `uunion` (lluu $ snd $ unzip ll')
        ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        xx' = apply xx ffr
        xxp' = historyRepasRed xx'
        xxrr' = apply xxrr ffr
        xxrrp' = historyRepasRed xxrr'
        gg = ff `funion` hh
        mm' = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      fromJust $ parametersSystemsBuilderTupleNoSumlayerRepa_1 xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = fromJust $ parametersSystemsPartitionerRepa_4 mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_1 qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (List.map (\((kk,_,_),a) -> (kk,a)) $
      parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_u wmax omax uu vv ff xx xxp xxrr xxrrp)
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = fromJust $ systemsTransformsTransformRepa uu tt
    hhvvr = historyRepasVectorVar
    apvvr = histogramRepaRedsVectorVar
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    ttpp = transformsPartition
    und = transformsUnderlying
    trans = histogramsSetVarsTransform_u
    unit qq = listsHistogram_u $ List.map (\ss -> (ss,1)) $ qq
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    uvars = systemsVars
    uunion = pairSystemsUnion
    lluu = listsSystem_u
    nnww = ValInt . toInteger
    maxr mm = if mm /= [] then (last $ sort $ snd $ unzip $ mm) else 0
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    subset = Set.isSubsetOf
    sgl = Set.singleton
    qqll = Set.toList
    vvqq = Set.fromList . V.toList

parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_2 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  Maybe (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_2 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 = Nothing
  | not (vvqq (hhvvr xx) `subset` uvars uu 
      && hhvvr xx == hhvvr xxrr && hhvvr xx == apvvr xxp && hhvvr xx == apvvr xxrrp 
      && vv `subset` vvqq (hhvvr xx)) = Nothing
  | otherwise = Just $ layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then 
        layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) else (uu,ff,mm) 
      where
        ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | 
               ((kk,bb),y1) <- buildfftup uu vv ff xx xxp xxrr xxrrp, 
               qq <- parter uu kk bb y1, (yy,pp) <- roller qq, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        hh = qqff $ llqq $ fst $ unzip ll'
        uu' = uu `uunion` (lluu $ snd $ unzip ll')
        ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        xx' = apply xx ffr
        xxp' = historyRepasRed xx'
        xxrr' = apply xxrr ffr
        xxrrp' = historyRepasRed xxrr'
        gg = ff `funion` hh
        mm' = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      fromJust $ parametersSystemsBuilderTupleNoSumlayerRepa xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = fromJust $ parametersSystemsPartitionerRepa mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (List.map (\((kk,_,_),a) -> (kk,a)) $
      parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_u wmax omax uu vv ff xx xxp xxrr xxrrp)
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = fromJust $ systemsTransformsTransformRepa uu tt
    hhvvr = historyRepasVectorVar
    apvvr = histogramRepaRedsVectorVar
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    ttpp = transformsPartition
    und = transformsUnderlying
    trans = histogramsSetVarsTransform_u
    unit qq = listsHistogram_u $ List.map (\ss -> (ss,1)) $ qq
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    cart uu vv = fromJust $ systemsVarsCartesian uu vv
    uvars = systemsVars
    uunion = pairSystemsUnion
    lluu = listsSystem_u
    nnww = ValInt . toInteger
    maxr mm = if mm /= [] then (last $ sort $ snd $ unzip $ mm) else 0
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    subset = Set.isSubsetOf
    sgl = Set.singleton
    qqll = Set.toList
    vvqq = Set.fromList . V.toList

parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u = 
   parametersSystemsLayererMaxRollTypeExcludedSelfHighestRepa_u MaximumRoll

parametersSystemsLayererMaxRollTypeExcludedSelfHighestRepa_u :: 
  MaxRollType -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaxRollTypeExcludedSelfHighestRepa_u 
  mroll wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then 
        layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) else (uu,ff,mm) 
      where
        ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | 
               ((kk,bb),y1) <- buildfftup uu vv ff xx xxp xxrr xxrrp, 
               qq <- parter uu kk bb y1, (yy,pp) <- roller qq, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        hh = qqff $ llqq $ fst $ unzip ll'
        uu' = uu `uunion` (lluu $ snd $ unzip ll')
        ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        xx' = apply xx ffr
        xxp' = historyRepasRed xx'
        xxrr' = apply xxrr ffr
        xxrrp' = historyRepasRed xxrr'
        gg = ff `funion` hh
        mm' = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_u xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = (if mroll == MaxRollByM then parametersSystemsPartitionerMaxRollByMRepa_u else parametersSystemsPartitionerRepa_u) 
                           mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (List.map (\((kk,_,_),a) -> (kk,a)) $
      parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_u wmax omax uu vv ff xx xxp xxrr xxrrp)
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    ttpp = transformsPartition
    und = transformsUnderlying
    trans = histogramsSetVarsTransform_u
    unit qq = listsHistogram_u $ List.map (\ss -> (ss,1)) $ qq
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    cart uu vv = systemsSetVarsSetStateCartesian_u uu vv
    uunion = pairSystemsUnion
    lluu = listsSystem_u
    nnww = ValInt . toInteger
    maxr mm = if mm /= [] then (last $ sort $ snd $ unzip $ mm) else 0
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    qqll = Set.toList

parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u_1 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then 
        layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) else (uu,ff,mm) 
      where
        ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | 
               ((kk,bb),y1) <- buildfftup uu vv ff xx xxp xxrr xxrrp, 
               qq <- parter uu kk bb y1, (yy,pp) <- roller qq, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        hh = qqff $ llqq $ fst $ unzip ll'
        uu' = uu `uunion` (lluu $ snd $ unzip ll')
        ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        xx' = apply xx ffr
        xxp' = historyRepasRed xx'
        xxrr' = apply xxrr ffr
        xxrrp' = historyRepasRed xxrr'
        gg = ff `funion` hh
        mm' = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleNoSumlayerRepa_u xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerRepa_u mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (List.map (\((kk,_,_),a) -> (kk,a)) $
      parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_u wmax omax uu vv ff xx xxp xxrr xxrrp)
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    ttpp = transformsPartition
    und = transformsUnderlying
    trans = histogramsSetVarsTransform_u
    unit qq = listsHistogram_u $ List.map (\ss -> (ss,1)) $ qq
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    cart uu vv = systemsSetVarsSetStateCartesian_u uu vv
    uunion = pairSystemsUnion
    lluu = listsSystem_u
    nnww = ValInt . toInteger
    maxr mm = if mm /= [] then (last $ sort $ snd $ unzip $ mm) else 0
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    qqll = Set.toList

parametersSystemsLayererMaxRollByMExcludedSelfHighestRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  Maybe (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaxRollByMExcludedSelfHighestRepa 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 = Nothing
  | not (vvqq (hhvvr xx) `subset` uvars uu 
      && hhvvr xx == hhvvr xxrr && hhvvr xx == apvvr xxp && hhvvr xx == apvvr xxrrp 
      && vv `subset` vvqq (hhvvr xx)) = Nothing
  | otherwise = Just $ parametersSystemsLayererMaxRollByMExcludedSelfHighestRepa_u 
      wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
  where
    hhvvr = historyRepasVectorVar
    apvvr = histogramRepaRedsVectorVar
    uvars = systemsVars
    vvqq = Set.fromList . V.toList
    subset = Set.isSubsetOf

parametersSystemsLayererMaxRollByMExcludedSelfHighestRepa_u :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaxRollByMExcludedSelfHighestRepa_u  = 
  parametersSystemsLayererMaxRollTypeExcludedSelfHighestRepa_u MaxRollByM

parametersSystemsDecomperHighestRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperHighestRepa wmax lmax xmax omax bmax mmax umax pmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || mult < 1 = Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1 seed
  where
    decomp uu zz f s
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1) (s + mult)
      | mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1) (s + mult)
      where
        aarr = ashuffle aa s mult
        (uur,ffr,nnr) = layerer uu aa aarr f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl (stateEmpty,ffr')
        mm = [(size bb,nn,ss,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let rrc = llsthis nn, let hhc = llfhis nn, let (_,ff) = last nn, ff /= fudEmpty,
                 ss <- qqll (cart uu (fder ff) `minus` dom (treesRoots yy)),
                 let xx = hhc `union` rrc `add` unit ss,
                 let bb = apply vv vv xx aa,
                 size bb > 0]
        (_,nn,ss,bb) = last $ sort mm
        bbrr = ashuffle bb s mult
        (uuc,ffc,nnc) = layerer uu bb bbrr f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [(ss,ffc')])
    layerer uu aa aarr f = fromJust $ 
        parametersSystemsLayererHighestRepa wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = hhhr uu $ aahh aa
        xxp = hrhx xx   
        xxrr = hhhr uu $ aahh aarr
        xxrrp = hrhx xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let (_,ff) = last ll in if ff == fudEmpty then init ll else ll
    llsthis = Set.fromList . List.map unit . fst . unzip
    llfhis = bigcup . Set.fromList . List.map fhis . snd . unzip
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    hrhx = historyRepasRed
    aahh aa = fromJust $ histogramsHistory aa
    hhaa hh = historiesHistogram hh
    hshuffle hh r = fromJust $ historiesShuffle hh (fromInteger r)
    ashuffle aa seed mult = let hh = aahh aa in 
                            foldl1 aadd [hhaa $ hshuffle hh (seed + r) | r <- [0..mult-1]]
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    bigcup = setSetsUnion
    dom = relationsDomain
    minus = Set.difference
    add qq x = Set.insert x qq
    qqll = Set.toList
    union = Set.union
    empty = Set.empty
    subset = Set.isSubsetOf
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperHighestFmaxRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer ->
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperHighestFmaxRepa wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 || fmax < 1 = Nothing
  | not (isint aa) || mult < 1 = Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1 seed
  where
    decomp uu zz f s
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1) (s + mult)
      | f > fmax || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1) (s + mult)
      where
        aarr = ashuffle aa s mult
        (uur,ffr,nnr) = layerer uu aa aarr f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl (stateEmpty,ffr')
        mm = [(size bb,nn,ss,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let rrc = llsthis nn, let hhc = llfhis nn, let (_,ff) = last nn, ff /= fudEmpty,
                 ss <- qqll (cart uu (fder ff) `minus` dom (treesRoots yy)),
                 let xx = hhc `union` rrc `add` unit ss,
                 let bb = apply vv vv xx aa,
                 size bb > 0]
        (_,nn,ss,bb) = last $ sort mm
        bbrr = ashuffle bb s mult
        (uuc,ffc,nnc) = layerer uu bb bbrr f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [(ss,ffc')])
    layerer uu aa aarr f = fromJust $ 
        parametersSystemsLayererHighestRepa wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = hhhr uu $ aahh aa
        xxp = hrhx xx   
        xxrr = hhhr uu $ aahh aarr
        xxrrp = hrhx xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let (_,ff) = last ll in if ff == fudEmpty then init ll else ll
    llsthis = Set.fromList . List.map unit . fst . unzip
    llfhis = bigcup . Set.fromList . List.map fhis . snd . unzip
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    hrhx = historyRepasRed
    aahh aa = fromJust $ histogramsHistory aa
    hhaa hh = historiesHistogram hh
    hshuffle hh r = fromJust $ historiesShuffle hh (fromInteger r)
    ashuffle aa seed mult = let hh = aahh aa in 
                            foldl1 aadd [hhaa $ hshuffle hh (seed + r) | r <- [0..mult-1]]
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    bigcup = setSetsUnion
    dom = relationsDomain
    minus = Set.difference
    add qq x = Set.insert x qq
    qqll = Set.toList
    union = Set.union
    empty = Set.empty
    subset = Set.isSubsetOf
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa wmax lmax xmax omax bmax mmax umax pmax mult seed uu vv aa =
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa wmax lmax xmax omax bmax mmax umax pmax 0 mult seed uu vv aa

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && vv `subset` qq) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (uur,ffr,nnr) = layerer uu aa f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl ((stateEmpty,ffr'),(aa, apply qq (fder ffr') (fhis ffr') aa))
        mm = [(a,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy))]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply qq qq (fhis ffb `add` unit ss) bb
        (uuc,ffc,nnc) = layerer uu cc f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(cc, apply qq (fder ffc') (fhis ffc') cc))])
    layerer uu aa f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxp = historyRepasRed xx   
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xxrrp = historyRepasRed xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    depends = fudsVarsDepends
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_4 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_4 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa =
    parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyRepa 
      wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa Set.empty

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_1 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1 seed
  where
    decomp uu zz f s
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1) (s + mult)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1) (s + mult)
      where
        (uur,ffr,nnr) = layerer uu aa s mult f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl (stateEmpty,ffr')
        mm = [(size bb,nn,ss,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let rrc = llsthis nn, let hhc = llfhis nn, let (_,ff) = last nn, ff /= fudEmpty,
                 ss <- qqll (cart uu (fder ff) `minus` dom (treesRoots yy)),
                 let xx = hhc `union` rrc `add` unit ss,
                 let bb = apply vv vv xx aa,
                 size bb > 0]
        (_,nn,ss,bb) = last $ sort mm
        (uuc,ffc,nnc) = layerer uu bb s mult f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [(ss,ffc')])
    layerer uu aa s mult f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        xxp = historyRepasRed xx   
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger i) | i <- [s..]]
        xxrrp = historyRepasRed xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let (_,ff) = last ll in if ff == fudEmpty then init ll else ll
    llsthis = Set.fromList . List.map unit . fst . unzip
    llfhis = bigcup . Set.fromList . List.map fhis . snd . unzip
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    bigcup = setSetsUnion
    dom = relationsDomain
    minus = Set.difference
    add qq x = Set.insert x qq
    qqll = Set.toList
    union = Set.union
    empty = Set.empty
    subset = Set.isSubsetOf
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_2 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_2 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1 seed
  where
    decomp uu zz f s
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1) (s + mult)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1) (s + mult)
      where
        (uur,ffr,nnr) = layerer uu aa s mult f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl ((stateEmpty,ffr'),(aa, apply vv (fder ffr') (fhis ffr') aa))
        mm = [(a,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall bb', a > 0, ss `notin` dom (dom (treesRoots yy))]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply vv vv (fhis ffb `add` unit ss) bb
        (uuc,ffc,nnc) = layerer uu cc s mult f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(cc, apply vv (fder ffc') (fhis ffc') cc))])
    layerer uu aa s mult f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        xxp = historyRepasRed xx   
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger i) | i <- [s..]]
        xxrrp = historyRepasRed xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa_1 wmax lmax xmax omax bmax mmax umax pmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || mult < 1 = Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1 seed
  where
    decomp uu zz f s
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1) (s + mult)
      | mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1) (s + mult)
      where
        aarr = ashuffle aa s mult
        (uur,ffr,nnr) = layerer uu aa aarr f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl (stateEmpty,ffr')
        mm = [(size bb,nn,ss,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let rrc = llsthis nn, let hhc = llfhis nn, let (_,ff) = last nn, ff /= fudEmpty,
                 ss <- qqll (cart uu (fder ff) `minus` dom (treesRoots yy)),
                 let xx = hhc `union` rrc `add` unit ss,
                 let bb = apply vv vv xx aa,
                 size bb > 0]
        (_,nn,ss,bb) = last $ sort mm
        bbrr = ashuffle bb s mult
        (uuc,ffc,nnc) = layerer uu bb bbrr f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [(ss,ffc')])
    layerer uu aa aarr f = fromJust $ parametersSystemsLayererMaximumRollExcludedSelfHighestRepa 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = hhhr uu $ aahh aa
        xxp = hrhx xx   
        xxrr = hhhr uu $ aahh aarr
        xxrrp = hrhx xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let (_,ff) = last ll in if ff == fudEmpty then init ll else ll
    llsthis = Set.fromList . List.map unit . fst . unzip
    llfhis = bigcup . Set.fromList . List.map fhis . snd . unzip
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    hrhx = historyRepasRed
    aahh aa = fromJust $ histogramsHistory aa
    hhaa hh = historiesHistogram hh
    hshuffle hh r = fromJust $ historiesShuffle hh (fromInteger r)
    ashuffle aa seed mult = let hh = aahh aa in 
                            foldl1 aadd [hhaa $ hshuffle hh (seed + r) | r <- [0..mult-1]]
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    bigcup = setSetsUnion
    dom = relationsDomain
    minus = Set.difference
    add qq x = Set.insert x qq
    qqll = Set.toList
    union = Set.union
    empty = Set.empty
    subset = Set.isSubsetOf
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa_2 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestRepa_2 wmax lmax xmax omax bmax mmax umax pmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1 seed
  where
    decomp uu zz f s
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1) (s + mult)
      | mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1) (s + mult)
      where
        aarr = ashuffle aa s mult
        (uur,ffr,nnr) = layerer uu aa aarr f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl (stateEmpty,ffr')
        mm = [(size bb,nn,ss,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let rrc = llsthis nn, let hhc = llfhis nn, let (_,ff) = last nn, ff /= fudEmpty,
                 ss <- qqll (cart uu (fder ff) `minus` dom (treesRoots yy)),
                 let xx = hhc `union` rrc `add` unit ss,
                 let bb = apply vv vv xx aa,
                 size bb > 0]
        (_,nn,ss,bb) = last $ sort mm
        bbrr = ashuffle bb s mult
        (uuc,ffc,nnc) = layerer uu bb bbrr f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [(ss,ffc')])
    layerer uu aa aarr f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = hhhr uu $ aahh aa
        xxp = hrhx xx   
        xxrr = hhhr uu $ aahh aarr
        xxrrp = hrhx xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let (_,ff) = last ll in if ff == fudEmpty then init ll else ll
    llsthis = Set.fromList . List.map unit . fst . unzip
    llfhis = bigcup . Set.fromList . List.map fhis . snd . unzip
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    hrhx = historyRepasRed
    aahh aa = fromJust $ histogramsHistory aa
    hhaa hh = historiesHistogram hh
    hshuffle hh r = fromJust $ historiesShuffle hh (fromInteger r)
    ashuffle aa seed mult = let hh = aahh aa in 
                            foldl1 aadd [hhaa $ hshuffle hh (seed + r) | r <- [0..mult-1]]
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    bigcup = setSetsUnion
    dom = relationsDomain
    minus = Set.difference
    add qq x = Set.insert x qq
    qqll = Set.toList
    union = Set.union
    empty = Set.empty
    subset = Set.isSubsetOf
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_3 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxRepa_3 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    decomp uu zz f
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (uur,ffr,nnr) = layerer uu aa f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl ((stateEmpty,ffr'),(aa, apply vv (fder ffr') (fhis ffr') aa))
        mm = [(a,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall bb', a > 0, ss `notin` dom (dom (treesRoots yy))]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply vv vv (fhis ffb `add` unit ss) bb
        (uuc,ffc,nnc) = layerer uu cc f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(cc, apply vv (fder ffc') (fhis ffc') cc))])
    layerer uu aa f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxp = historyRepasRed xx   
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xxrrp = historyRepasRed xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> Set.Set Variable -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyRepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa ll
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && vv `subset` qq && ll `subset` qq) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (uur,ffr,nnr) = layerer uu aa f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl ((stateEmpty,ffr'),(aa, apply qq (fder ffr' `cup` ll) (fhis ffr') aa))
        mm = [(b,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy)),
                 let b = if Set.null ll then fromRational a else fromRational a * entropy (bb' `mul` unit ss `red` ll), 
                 b > 0]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply qq qq (fhis ffb `add` unit ss) bb
        (uuc,ffc,nnc) = layerer uu cc f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(cc, apply qq (fder ffc') (fhis ffc') cc))])
    layerer uu aa f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxp = historyRepasRed xx   
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xxrrp = historyRepasRed xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    entropy = histogramsEntropy
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    cup = Set.union
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> Set.Set Variable -> Set.Set Variable -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelRepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa ll lld
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && vv `subset` qq && ll `subset` qq) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (uur,ffr,nnr) = layerer uu aa f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr (ndep lld ffr kkr) else fudEmpty
        zzr = tsgl ((stateEmpty,ffr'),(aa, apply qq (fder ffr' `cup` ll) (fhis ffr') aa))
        mm = [(b,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy)),
                 let b = if Set.null ll then fromRational a else fromRational a * entropy (bb' `mul` unit ss `red` ll), 
                 b > 0]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply qq qq (fhis ffb `add` unit ss) bb
        (uuc,ffc,nnc) = layerer uu cc f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc (ndep lld ffc kkc) else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` 
                (nn List.++ [((ss,ffc'),(cc, apply qq (fder ffc' `cup` ll) (fhis ffc') cc))])
    layerer uu aa f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxp = historyRepasRed xx   
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xxrrp = historyRepasRed xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fvars = fudsVars
    fhis = fudsSetHistogram
    entropy = histogramsEntropy
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    cap = Set.intersection
    cup = Set.union
    flip = List.map (\(a,b) -> (b,a))

systemsDecompFudsHistoryRepasAlignmentContentShuffleSummation_u :: 
  Integer -> Integer -> System -> DecompFud -> HistoryRepa -> (Double,Double)
systemsDecompFudsHistoryRepasAlignmentContentShuffleSummation_u mult seed uu df aa =
    Set.fold scalgn (0,0) $ treesElements $ apply mult seed uu df aa
  where
    scalgn ((_,ff),(hr,hrxx)) (a,ad) = (a + b, ad + b/(u ** (1/m)))
      where
        u = fromIntegral (vol uu (vars aa))
        m = fromIntegral (Set.size (vars aa))
        aa = araa uu (hr `hrred` fder ff)
        bb = resize (size aa) (araa uu (hrxx `hrred` fder ff))
        b = algn aa - algn bb
    apply = systemsDecompFudsHistoryRepasMultiplyWithShuffle
    hrred hh vv = setVarsHistoryRepasReduce 1 vv hh
    araa uu rr = fromJust $ systemsHistogramRepasHistogram uu rr
    fder = fudsDerived
    algn = histogramsAlignment
    vars = histogramsVars
    size = histogramsSize
    resize z aa = if z > 0 then (fromJust $ histogramsResize z aa) else empty
    empty = histogramEmpty
    vol uu vv = fromJust $ systemsVarsVolume uu vv

systemsDecompFudsHistoryRepasTreeAlignmentContentShuffleSummation_u :: 
  Integer -> Integer -> System -> DecompFud -> HistoryRepa -> Tree ((State,Fud),(Int,(Double,Double)))
systemsDecompFudsHistoryRepasTreeAlignmentContentShuffleSummation_u mult seed uu df aa =
    funcsTreesMap scalgn $ apply mult seed uu df aa
  where
    scalgn ((ss,ff),(hr,hrxx)) = ((ss,ff),(hrsize hr, (b, b/(u ** (1/m)))))
      where
        u = fromIntegral (vol uu (vars aa))
        m = fromIntegral (Set.size (vars aa))
        aa = araa uu (hr `hrred` fder ff)
        bb = resize (size aa) (araa uu (hrxx `hrred` fder ff))
        b = algn aa - algn bb
    apply = systemsDecompFudsHistoryRepasMultiplyWithShuffle
    hrred hh vv = setVarsHistoryRepasReduce 1 vv hh
    araa uu rr = fromJust $ systemsHistogramRepasHistogram uu rr
    hrsize = historyRepasSize
    fder = fudsDerived
    algn = histogramsAlignment
    vars = histogramsVars
    size = histogramsSize
    resize z aa = if z > 0 then (fromJust $ histogramsResize z aa) else empty
    empty = histogramEmpty
    vol uu vv = fromJust $ systemsVarsVolume uu vv

systemsDecompFudsHistoryRepasAlgnDensPerSizesStripped_u :: 
  Integer -> Integer -> System -> DecompFud -> HistoryRepa -> Double -> DecompFud
systemsDecompFudsHistoryRepasAlgnDensPerSizesStripped_u mult seed uu df aa r =
    zzdf $ llzz $ List.map (fst . unzip . takeWhile (\(_,adz) -> adz > r)) $ 
      zzll $ funcsTreesMap (\((ss,ff),(zc, (a, ad))) -> ((ss,ff),ad / fromIntegral zc)) $ 
        sumtree mult seed uu df aa
  where
    sumtree = systemsDecompFudsHistoryRepasTreeAlignmentContentShuffleSummation_u
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    zzll zz = Set.toList $ treesPaths zz
    llzz ll = pathsTree $ Set.fromList ll

parametersSystemsBuilderTupleLevelNoSumlayerRepa_u :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleLevelNoSumlayerRepa_u xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx =
    fst $ buildfftup xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  where
    buildfftup = parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui

parametersSystemsBuilderTupleLevelNoSumlayerRepa_u_1 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleLevelNoSumlayerRepa_u_1 xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  | ff == fudEmpty = V.toList $ topd (bmax `div` mmax) $ buildb vv (init vv) V.empty
  | otherwise = 
      V.toList $ topd (bmax `div` mmax) $ buildb (fvars ff `minus` fvars ffg `union` vv) (init (fder ff)) V.empty
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vv = vvg `union` fder ffg
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn = if (not (V.null mm)) then buildb ww mm (nn V.++ mm) else (final nn) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_u :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_u xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx =
    fst $ buildfftup xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  where
    buildfftup = parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_ui

parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_u_1 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)]
parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_u_1 xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  | vv' == Set.empty = []
  | ff == fudEmpty = V.toList $ topd (bmax `div` mmax) $ buildb vv' (init vv') V.empty
  | otherwise = 
      V.toList $ topd (bmax `div` mmax) $ buildb (fvars ff `minus` fvars ffg `union` vv') (init (fder ff)) V.empty
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vv = vvg `union` fder ffg
    vv' = meff hhx vv
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn = if (not (V.null mm)) then buildb ww mm (nn V.++ mm) else (final nn) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    meff hhx vv = Set.fromList [v | v <- qqll vv, let i = hhxv hhx Map.! v, 
      length (List.filter (/=0) (UV.toList (hhxi hhx V.! i))) > 1]
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    hhxv = histogramRepaRedsMapVarInt
    hhxi = histogramRepaRedsVectorArray
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (res (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (res (topd (bmax `div` mmax) x1), s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    vv = vvg `union` fder ffg
    (xc,sc) = cross xmax omax vv hh hhx hhrr hhrrx
    yy = (fvars ff `minus` fvars ffg `union` vv)
    (xa,sa) = append xmax omax yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x0,s0) = buildb vv xc xc sc
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append xmax omax ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    res xx = [((jj, (bbv,ffv,ssv)), a1-b1) | jj <- vvll xx,
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,_,b1,_] = UV.toList ssv]
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    cross = parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvll = V.toList
    qqvv = V.fromList . Set.toList

parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_1 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_1 xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (V.toList (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (V.toList (topd (bmax `div` mmax) x1), s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vv = vvg `union` fder ffg
    (x0,s0) = buildb vv (init vv) V.empty 0
    (x1,s1) = buildb (fvars ff `minus` fvars ffg `union` vv) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn s2 = 
        if (not (V.null mm)) then buildb ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
        mm = top omax x2
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce_3
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_2 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_2 xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (V.toList (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (V.toList (topd (bmax `div` mmax) x1), s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    vv = vvg `union` fder ffg
    (x0,s0) = buildb vv (init vv) V.empty 0
    (x1,s1) = buildb (fvars ff `minus` fvars ffg `union` vv) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    buildb ww qq nn s2 = 
        if (not (V.null mm)) then buildb ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
        mm = top omax x2
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_3 :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui_3 xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  | ff == fudEmpty = (V.toList (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (V.toList (topd (bmax `div` mmax) x1), s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    vv = vvg `union` fder ffg
    xc = initc vv
    (x0,s0) = buildb vv xc xc (toInteger (Set.size vv * (Set.size vv - 1) `div` 2))
    (x1,s1) = buildb (fvars ff `minus` fvars ffg `union` vv) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),((sgl w, (hvempty, hvempty, UV.empty)),0)) | w <- qqll vv]
    initc vv = 
      let pp = cross xmax omax vv hh hhx hhrr hhrrx in
        V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |              
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    buildb ww qq nn s2 = 
        if (not (V.null mm)) then buildb ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,((kk,_),_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [((a1-a2-b1+b2, -b1+b2, -u),((jj, (bbv,ffv,ssv)), a1-b1)) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
        mm = top omax x2
    final = V.filter (\(_,((kk,_),_)) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    cross = parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u_1
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_ui :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, (HistogramRepaVec, HistogramRepaVec, UV.Vector Double)),Double)],Integer)
parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_ui xmax omax bmax mmax uu vvg ffg ff hh hhx hhrr hhrrx
  | Set.size vv' < 2 = ([],0)
  | ff == fudEmpty = (res (topd (bmax `div` mmax) x0), s0) 
  | otherwise = (res (topd (bmax `div` mmax) x1), s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    vv = vvg `union` fder ffg
    vv' = meff hhx vv
    (xc,sc) = cross xmax omax vv' hh hhx hhrr hhrrx
    yy = (fvars ff `minus` fvars ffg `union` vv')
    (xa,sa) = append xmax omax yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x0,s0) = buildb vv' xc xc sc
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append xmax omax ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    res xx = [((jj, (bbv,ffv,ssv)), a1-b1) | jj <- vvll xx,
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,_,b1,_] = UV.toList ssv]
    meff hhx vv = Set.fromList [v | v <- qqll vv, let i = hhxv hhx Map.! v, 
      length (List.filter (/=0) (UV.toList (hhxi hhx V.! i))) > 1]
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepaStorablesReduce
    cross = parametersSetVarsHistoryRepasSetSetVarsAlignedTop_u
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedTop_u
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    hhxv = histogramRepaRedsMapVarInt
    hhxi = histogramRepaRedsVectorArray
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    topd amax mm = snd $ V.unzip $ vectorPairsTop (fromInteger amax) mm
    add xx x = x `Set.insert` xx
    union = Set.union
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvll = V.toList
    qqvv = V.fromList . Set.toList

parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_u :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_u wmax omax uu vv ffg ff hh hhx hhrr hhrrx = 
    fst $ buildffdervar wmax omax uu vv ffg ff hh hhx hhrr hhrrx
  where
    buildffdervar = parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_ui 

parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_ui :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, HistogramRepa, HistogramRepa), Double)],Integer)
parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_ui wmax omax uu vv ffg ff hh hhx hhrr hhrrx
  | yy == Set.empty = ([],0)
  | otherwise = (res (maxd x1),s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    vv' = vv `union` fvars ffg
    cc = Set.fromList [(w,u) | w <- Set.toList (fvars ff `minus` vv'), let gg = depends ff w, 
                               u <- Set.toList (fvars gg `minus` vv'), u /= w]
    yy = fvars ff `minus` vv'
    (xa,sa) = append wmax omax cc yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append wmax omax cc ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    res xx = [((jj, bb, bbrr), (a-b)/c) | jj <- vvll xx, let u = vol uu jj, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepaStorablesReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedExcludeHiddenDenseTop_u
    depends ff w = fudsVarsDepends ff (Set.singleton w)
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    maxd mm = snd $ V.unzip $ vectorPairsTop 1 mm
    union = Set.union
    minus = Set.difference
    sgl = Set.singleton
    vvll = V.toList
    qqvv = V.fromList . Set.toList

parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_u :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_u wmax omax uu vv ffg ff hh hhx hhrr hhrrx = 
    fst $ buildffdervar wmax omax uu vv ffg ff hh hhx hhrr hhrrx
  where
    buildffdervar = parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui 

parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_u_1 :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  [((Set.Set Variable, HistogramRepa, HistogramRepa), Double)]
parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_u_1 wmax omax uu vv ffg ff hh hhx hhrr hhrrx = 
  V.toList $ maxfst $ buildd (fvars ff `minus` vv `minus` fvars ffg) (init (fder ff)) V.empty
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = V.fromListN (card vv) [((0,0,0),(sgl w, hempty, hempty)) | w <- qqll vv]
    buildd ww qq nn = if (not (V.null mm)) then buildd ww mm (nn V.++ mm) else (final nn) 
      where
        pp = llqq [jj | (_,(kk,_,_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ V.fromListN (card pp) [(((a-b)/c,-b/c,-u),(jj, bb, bbrr)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax,
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    final = V.filter (\(_,(kk,_,_)) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    maxfst mm = V.map (\((a,_,_),x) -> (x,a)) $ vectorPairsTop 1 mm
    add xx x = x `Set.insert` xx
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, HistogramRepa, HistogramRepa), Double)],Integer)
parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui wmax omax uu vv ffg ff hh hhx hhrr hhrrx
  | yy == Set.empty = ([],0)
  | otherwise = (res (maxd x1),s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    yy = fvars ff `minus` vv `minus` fvars ffg
    (xa,sa) = append wmax omax yy (qqvv (Set.map sgl (fder ff))) hh hhx hhrr hhrrx
    (x1,s1) = buildb yy xa xa sa
    buildb ww qq nn sn
      | V.null qq = (nn,sn) 
      | not (V.null mm) = buildb ww mm (nn V.++ mm) (sn + sm)
      | otherwise = (nn,sn) 
      where
        (mm,sm) = append wmax omax ww (snd $ V.unzip qq) hh hhx hhrr hhrrx
    res xx = [((jj, bb, bbrr), (a-b)/c) | jj <- vvll xx, let u = vol uu jj, 
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepaStorablesReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    append = parametersSetVarsSetSetVarsHistoryRepasSetSetVarsAlignedDenseTop_u
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    maxd mm = snd $ V.unzip $ vectorPairsTop 1 mm
    minus = Set.difference
    sgl = Set.singleton
    vvll = V.toList
    qqvv = V.fromList . Set.toList

parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui_1 :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, HistogramRepa, HistogramRepa), Double)],Integer)
parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui_1 wmax omax uu vv ffg ff hh hhx hhrr hhrrx = 
    (V.toList (maxfst x1),s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    (x1,s1) =  buildd (fvars ff `minus` vv `minus` fvars ffg) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),(sgl w, hempty, hempty)) | w <- qqll vv]
    buildd ww qq nn s2 = 
        if (not (V.null mm)) then buildd ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,(kk,_,_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [(((a-b)/c,-b/c,-u),(jj, bb, bbrr)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax,
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
        mm = top omax x2
    final = V.filter (\(_,(kk,_,_)) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepasReduce_3
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    maxfst mm = V.map (\((a,_,_),x) -> (x,a)) $ vectorPairsTop 1 mm
    add xx x = x `Set.insert` xx
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui_2 :: 
  Integer -> Integer -> System -> Set.Set Variable -> Fud -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  ([((Set.Set Variable, HistogramRepa, HistogramRepa), Double)],Integer)
parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerIncludeHiddenRepa_ui_2 wmax omax uu vv ffg ff hh hhx hhrr hhrrx = 
    (V.toList (maxfst x1),s1)
  where
    Z :. _ :. z = extent $ historyRepasArray hh
    Z :. _ :. zrr = extent $ historyRepasArray hhrr
    f = (fromIntegral z)/(fromIntegral zrr)
    vshh = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hh))) :: SV.Vector CLLong
    vshhrr = SV.unsafeCast (UV.convert (R.toUnboxed (historyRepasArray hhrr))) :: SV.Vector CLLong
    (x1,s1) =  buildd (fvars ff `minus` vv `minus` fvars ffg) (init (fder ff)) V.empty 0
    init vv = V.fromListN (card vv) [((0,0,0),(sgl w, hempty, hempty)) | w <- qqll vv]
    buildd ww qq nn s2 = 
        if (not (V.null mm)) then buildd ww mm (nn V.++ mm) (s2 + toInteger (V.length x2)) else ((final nn),s2) 
      where
        pp = llqq [jj | (_,(kk,_,_)) <- V.toList qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        x2 = V.fromListN (card pp) [(((a-b)/c,-b/c,-u),(jj, bb, bbrr)) |
          jj <- qqll pp, let u = vol uu jj, u <= wmax,
          let bb = reduce 1 jj hh vshh, let bbrr = reduce f jj hhrr vshhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let u' = fromIntegral u, let m = fromIntegral (Set.size jj),
          let a = sumfacln bb - sumfacln bbx, let b = sumfacln bbrr - sumfacln bbrrx, let c = u' ** (1/m)]
        mm = top omax x2
    final = V.filter (\(_,(kk,_,_)) -> card kk > 1) 
    sumfacln (HistogramRepa _ _ rr) = UV.sum $ UV.map (\x -> logGamma (x + 1)) (toUnboxed rr)
    reduce = setVarsHistoryRepaStorablesReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hempty = histogramRepaEmpty
    fder = fudsDerived
    fvars = fudsVars
    vol uu vv = systemsSetVarsVolume_u uu vv
    top amax mm = vectorPairsTop (fromInteger amax) mm
    maxfst mm = V.map (\((a,_,_),x) -> (x,a)) $ vectorPairsTop 1 mm
    add xx x = x `Set.insert` xx
    minus = Set.difference
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton

parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer -> Integer ->
  (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u = 
  parametersSystemsLayererLevelMaxRollTypeExcludedSelfHighestRepa_u MaximumRoll

parametersSystemsLayererLevelMaxRollTypeExcludedSelfHighestRepa_u :: 
  MaxRollType -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer -> Integer ->
  (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererLevelMaxRollTypeExcludedSelfHighestRepa_u 
  mroll wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx xxp xxrr xxrrp f g = 
    layer uu fudEmpty [] xx xxp xxrr xxrrp 1
  where
    layer uu ff mm xx xxp xxrr xxrrp l = 
      if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then 
        layer uu' gg mm' xx' xxp' xxrr' xxrrp' (l+1) else (uu,ff,mm) 
      where
        ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | 
               ((kk,bb),y1) <- buildfftup uu vvg ffg ff xx xxp xxrr xxrrp, 
               qq <- parter uu kk bb y1, (yy,pp) <- roller qq, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarPair (VarInt f, VarInt g), VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        hh = qqff $ llqq $ fst $ unzip ll'
        uu' = uu `uunion` (lluu $ snd $ unzip ll')
        ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        xx' = apply xx ffr
        xxp' = historyRepasRed xx'
        xxrr' = apply xxrr ffr
        xxrrp' = historyRepasRed xxrr'
        gg = ff `funion` hh `funion` depends ffg (fund hh)
        mm' = buildffdervar uu' vvg ffg gg xx' xxp' xxrr' xxrrp'
    buildfftup uu vvg ffg ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_u xmax omax bmax mmax uu vvg ffg ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = (if mroll == MaxRollByM then parametersSystemsPartitionerMaxRollByMRepa_u else parametersSystemsPartitionerRepa_u) 
                           mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa qq
    buildffdervar uu vv ffg ff xx xxp xxrr xxrrp = (List.map (\((kk,_,_),a) -> (kk,a)) $
      parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_u 
        wmax omax uu vv ffg ff xx xxp xxrr xxrrp)
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    depends = fudsVarsDepends
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    ttpp = transformsPartition
    und = transformsUnderlying
    fund = fudsUnderlying
    trans = histogramsSetVarsTransform_u
    unit qq = listsHistogram_u $ List.map (\ss -> (ss,1)) $ qq
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    cart uu vv = systemsSetVarsSetStateCartesian_u uu vv
    uunion = pairSystemsUnion
    lluu = listsSystem_u
    nnww = ValInt . toInteger
    maxr mm = if mm /= [] then (last $ sort $ snd $ unzip $ mm) else 0
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    qqll = Set.toList

parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer -> Integer ->
  (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u_1 
  wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx xxp xxrr xxrrp f g = 
    layer uu fudEmpty [] xx xxp xxrr xxrrp 1
  where
    layer uu ff mm xx xxp xxrr xxrrp l = 
      if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then 
        layer uu' gg mm' xx' xxp' xxrr' xxrrp' (l+1) else (uu,ff,mm) 
      where
        ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | 
               ((kk,bb),y1) <- buildfftup uu vvg ffg ff xx xxp xxrr xxrrp, 
               qq <- parter uu kk bb y1, (yy,pp) <- roller qq, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarPair (VarInt f, VarInt g), VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        hh = qqff $ llqq $ fst $ unzip ll'
        uu' = uu `uunion` (lluu $ snd $ unzip ll')
        ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        xx' = apply xx ffr
        xxp' = historyRepasRed xx'
        xxrr' = apply xxrr ffr
        xxrrp' = historyRepasRed xxrr'
        gg = ff `funion` hh `funion` depends ffg (fund hh)
        mm' = buildffdervar uu' vvg ffg gg xx' xxp' xxrr' xxrrp'
    buildfftup uu vvg ffg ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleLevelNoSumlayerRepa_u xmax omax bmax mmax uu vvg ffg ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerRepa_u mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa qq
    buildffdervar uu vv ffg ff xx xxp xxrr xxrrp = (List.map (\((kk,_,_),a) -> (kk,a)) $
      parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_u 
        wmax omax uu vv ffg ff xx xxp xxrr xxrrp)
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    depends = fudsVarsDepends
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fund = fudsUnderlying
    ttpp = transformsPartition
    und = transformsUnderlying
    trans = histogramsSetVarsTransform_u
    unit qq = listsHistogram_u $ List.map (\ss -> (ss,1)) $ qq
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    cart uu vv = systemsSetVarsSetStateCartesian_u uu vv
    uunion = pairSystemsUnion
    lluu = listsSystem_u
    nnww = ValInt . toInteger
    maxr mm = if mm /= [] then (last $ sort $ snd $ unzip $ mm) else 0
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    qqll = Set.toList

parametersSystemsLayererLevelMaxRollByMExcludedSelfHighestRepa_u :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer -> Integer ->
  (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererLevelMaxRollByMExcludedSelfHighestRepa_u =
  parametersSystemsLayererLevelMaxRollTypeExcludedSelfHighestRepa_u MaxRollByM

parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Histogram -> Tree (Integer, Set.Set Variable, Fud) -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxRepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg =
    parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelRepa 
      lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg Set.empty Set.empty

parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Histogram -> Tree (Integer, Set.Set Variable, Fud) -> 
  Set.Set Variable -> Set.Set Variable -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelRepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg ll lld
  | lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && ll `subset` qq) = Nothing
  | not (okLevel zzg) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && ffr == fudEmpty = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (uur,ffr,_) = level uu aa zzg f 1
        zzr = tsgl ((stateEmpty,ffr),(aa, apply qq (fder ffr `cup` ll) (fhis ffr) aa))
        mm = [(b,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy)),
                 let b = if Set.null ll then fromRational a else fromRational a * entropy (bb' `mul` unit ss `red` ll), 
                 b > 0]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply qq qq (fhis ffb `add` unit ss) bb
        (uuc,ffc,_) = level uu cc zzg f 1
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc),(cc, apply qq (fder ffc `cup` ll) (fhis ffc) cc))])
    level uu aa (Tree ttg) f g = foldl next (uu,fudEmpty,g) (Map.toList ttg)
      where       
        next (uu,ff,g) ((wmaxg,vvg,ffg),xxg) = (uu',ff `funion` gg',gh+1)
          where
            (uuh,ffh,gh) = level uu aa xxg f g
            (uu',gg,nn) = layerer wmaxg uuh vvg (ffg `funion` ffh) aa f gh
            (a,kk) = maxd nn
            gg' = if a > repaRounding then depends gg (ndep lld gg kk) else fudEmpty
    layerer wmax uu vvg ffg aa f g = parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx' xxp' xxrr' xxrrp' f g
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xx' = applyhr uu ffg xx
        xxp' = historyRepasRed xx'   
        xxrr' = applyhr uu ffg xxrr
        xxrrp' = historyRepasRed xxrr'   
    okLevel zzg = and [wmaxg >= 0 && vvg `subset` vars aa && fvars ffg `subset` uvars uu && fund ffg `subset` vars aa |
                       (wmaxg,vvg,ffg) <- Set.toList (treesElements zzg)]
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fvars = fudsVars
    fund = fudsUnderlying
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fhis = fudsSetHistogram
    entropy = histogramsEntropy
    applyhr uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    cap = Set.intersection
    cup = Set.union
    llvv = V.fromList
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelGoodnessRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Histogram -> Tree (Integer, Set.Set Variable, Fud) -> 
  Set.Set Variable -> Set.Set Variable -> 
  (System -> Histogram -> Histogram -> Fud -> Double) -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelGoodnessRepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg ll lld good
  | lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && ll `subset` qq) = Nothing
  | not (okLevel zzg) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && ffr == fudEmpty = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (_,(uur,ffr)) = best uu aa zzg f 1
        zzr = tsgl ((stateEmpty,ffr),(aa, apply qq (fder ffr `cup` ll) (fhis ffr) aa))
        mm = [(b,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy)),
                 let b = if Set.null ll then fromRational a else fromRational a * entropy (bb' `mul` unit ss `red` ll), 
                 b > 0]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply qq qq (fhis ffb `add` unit ss) bb
        (_,(uuc,ffc)) = best uu cc zzg f 1
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc),(cc, apply qq (fder ffc `cup` ll) (fhis ffc) cc))])
    best uu aa (Tree ttg) f h
      | qq /= [] && gg' /= fudEmpty && g' > g = (g', (uu'',gg'))
      | qq /= [] = (g, (uu',gg))
      | otherwise = (0,(uu,fudEmpty))
      where       
        aarr = ashuffle aa seed mult
        qq = [(good uu' aa aarr gg, (gg,uu'), xxg) | ((wmaxg,vvg,ffg),xxg) <- Map.toList ttg, 
               let (uu',ff,nn) = layerer wmaxg uu vvg ffg aa f h,
               let (a,kk) = maxd nn, a > repaRounding, let gg = depends ff (ndep lld ff kk)]
        (g, (gg,uu'), xxg) = last $ sort qq
        (g', (uu'',gg')) = best uu' aa xxg f (h+1)
    layerer wmax uu vvg ffg aa f g = parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx' xxp' xxrr' xxrrp' f g
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xx' = applyhr uu ffg xx
        xxp' = historyRepasRed xx'   
        xxrr' = applyhr uu ffg xxrr
        xxrrp' = historyRepasRed xxrr'   
    okLevel zzg = and [wmaxg >= 0 && vvg `subset` vars aa && fvars ffg `subset` uvars uu && fund ffg `subset` vars aa |
                       (wmaxg,vvg,ffg) <- Set.toList (treesElements zzg)]
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fvars = fudsVars
    fund = fudsUnderlying
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fhis = fudsSetHistogram
    hhaa hh = historiesHistogram hh
    hshuffle hh r = fromJust $ historiesShuffle hh (fromInteger r)
    ashuffle aa seed mult = let hh = aahh aa in 
                            resize (size aa) $ foldl1 aadd [hhaa $ hshuffle hh (seed + r) | r <- [0..mult-1]]
    entropy = histogramsEntropy
    applyhr uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    aall = histogramsList
    resize z aa = fromJust $ histogramsResize z aa
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    cap = Set.intersection
    cup = Set.union
    llvv = V.fromList
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsBuilderLabelTupleRepa :: 
  Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Set.Set Variable -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed ->   
  Maybe [(Set.Set Variable, Double)]
parametersSystemsBuilderLabelTupleRepa xmax omax bmax mmax uu vv vvl hh hhx hhrr hhrrx
  | xmax < 0 || omax < 0 || mmax < 1 || bmax < mmax = Nothing
  | z == 0 || zrr == 0 = Nothing
  | not (vvqq vhh `subset` uvars uu && vhh == vhhrr && vhh == vhhx && vhhx == vhhrrx && vv `subset` vvqq vhh) = Nothing
  | otherwise = 
      Just $ top (bmax `div` mmax) $ buildb vv (init vvl) []
  where
    HistoryRepa vhh _ _ aa = hh
    HistogramRepaRed vhhx _ _ _ = hhx
    HistoryRepa vhhrr _ _ aarr = hhrr
    HistogramRepaRed vhhrrx _ _ _ = hhrrx
    Z :. _ :. z = extent aa
    Z :. _ :. zrr = extent aarr
    f = (fromIntegral z)/(fromIntegral zrr)
    init vv = [(sgl w, 0) | w <- qqll vv]
    buildb ww qq nn = if mm /= [] then buildb ww mm (nn List.++ mm) else (final nn) 
      where
        pp = llqq [jj | (kk,_) <- qq, w <- qqll (ww `minus` kk), let jj = kk `add` w]
        mm = top omax $ [(jj, a1-a2-b1+b2) |
          jj <- qqll pp, let u = vol uu jj, u <= xmax,
          let bb = reduce 1 jj hh, let bbrr = reduce f jj hhrr,
          let bbx = xind z (hhx `xred` jj), let bbrrx = xind z (hhrrx `xred` jj), 
          let bbv = vrrrrv z $ V.fromListN 4 [bb, bbx, bbrr, bbrrx], 
          let ffv = rrvffv bbv, let ssv = rrvsum ffv,
          let [a1,a2,b1,b2] = UV.toList ssv]
    final = List.filter (\(kk,_) -> card kk > 1) 
    fder = fudsDerived
    fvars = fudsVars
    rrvffv = histogramRepaVecsFaclnsRepaVecs
    rrvsum = histogramRepaVecsSum
    reduce = setVarsHistoryRepasReduce
    xred hhx vv = setVarsHistogramRepaRedsRed vv hhx
    xind x hhx = histogramRepaRedsIndependent (fromIntegral x) hhx
    hvempty = histogramRepaVecEmpty
    vrrrrv x = vectorHistogramRepasHistogramRepaVec_u (fromIntegral x)
    vol uu vv = fromJust $ systemsVarsVolume uu vv
    uvars = systemsVars
    top amax mm = flip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    topd amax mm = snd $ unzip $ take (fromInteger amax) $ reverse $ sort $ flip $ mm
    flip = List.map (\(a,b) -> (b,a))
    sumlayer ff kk = sum [layer ff (sgl w) | w <- qqll kk]
    layer = fudsSetVarsLayer
    add xx x = x `Set.insert` xx
    minus = Set.difference
    subset = Set.isSubsetOf
    card = Set.size
    qqll :: forall a. Set.Set a -> [a]
    qqll = Set.toList
    llqq :: forall a. (Ord a) => [a] -> Set.Set a
    llqq = Set.fromList
    sgl = Set.singleton
    vvqq = Set.fromList . V.toList

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> Set.Set Variable -> Set.Set Variable -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelRepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa ll lld
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && vv `subset` qq && ll `subset` qq) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (uur,ffr,nnr) = layerer uu aa f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr (ndep lld ffr kkr) else fudEmpty
        zzr = tsgl ((stateEmpty,ffr'),(aa, apply qq (fder ffr' `cup` ll) (fhis ffr') aa))
        mm = [(b,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy)),
                 let b = if Set.null ll then a else a - aamax (bb' `mul` unit ss `red` ll), 
                 b > 0]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply qq qq (fhis ffb `add` unit ss) bb
        (uuc,ffc,nnc) = layerer uu cc f
        (ac,kkc) = maxd nnc
        ffc' = if ac > repaRounding then depends ffc (ndep lld ffc kkc) else fudEmpty
        zzc = pathsTree $ treesPaths zz `add` 
                (nn List.++ [((ss,ffc'),(cc, apply qq (fder ffc' `cup` ll) (fhis ffc') cc))])
    layerer uu aa f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxp = historyRepasRed xx   
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xxrrp = historyRepasRed xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    ffqq = fudsSetTransform
    fder = fudsDerived
    fvars = fudsVars
    fhis = fudsSetHistogram
    aamax aa = if size aa > 0 then (last $ sort $ snd $ unzip $ aall aa) else 0
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    cap = Set.intersection
    cup = Set.union
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Histogram -> Tree (Integer, Set.Set Variable, Fud) -> 
  Set.Set Variable -> Set.Set Variable -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelRepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg ll lld
  | lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && ll `subset` qq) = Nothing
  | not (okLevel zzg) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && ffr == fudEmpty = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (uur,ffr,_) = level uu aa zzg f 1
        zzr = tsgl ((stateEmpty,ffr),(aa, apply qq (fder ffr `cup` ll) (fhis ffr) aa))
        mm = [(b,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy)),
                 let b = if Set.null ll then a else a - aamax (bb' `mul` unit ss `red` ll), 
                 b > 0]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply qq qq (fhis ffb `add` unit ss) bb
        (uuc,ffc,_) = level uu cc zzg f 1
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc),(cc, apply qq (fder ffc `cup` ll) (fhis ffc) cc))])
    level uu aa (Tree ttg) f g = foldl next (uu,fudEmpty,g) (Map.toList ttg)
      where       
        next (uu,ff,g) ((wmaxg,vvg,ffg),xxg) = (uu',ff `funion` gg',gh+1)
          where
            (uuh,ffh,gh) = level uu aa xxg f g
            (uu',gg,nn) = layerer wmaxg uuh vvg (ffg `funion` ffh) aa f gh
            (a,kk) = maxd nn
            gg' = if a > repaRounding then depends gg (ndep lld gg kk) else fudEmpty
    layerer wmax uu vvg ffg aa f g = parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx' xxp' xxrr' xxrrp' f g
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xx' = applyhr uu ffg xx
        xxp' = historyRepasRed xx'   
        xxrr' = applyhr uu ffg xxrr
        xxrrp' = historyRepasRed xxrr'   
    okLevel zzg = and [wmaxg >= 0 && vvg `subset` vars aa && fvars ffg `subset` uvars uu && fund ffg `subset` vars aa |
                       (wmaxg,vvg,ffg) <- Set.toList (treesElements zzg)]
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fvars = fudsVars
    fund = fudsUnderlying
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fhis = fudsSetHistogram
    aamax aa = if size aa > 0 then (last $ sort $ snd $ unzip $ aall aa) else 0
    applyhr uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    cap = Set.intersection
    cup = Set.union
    llvv = V.fromList
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelGoodnessRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Histogram -> Tree (Integer, Set.Set Variable, Fud) -> 
  Set.Set Variable -> Set.Set Variable -> 
  (System -> Histogram -> Histogram -> Fud -> Double) -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelGoodnessRepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg ll lld good
  | lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && ll `subset` qq) = Nothing
  | not (okLevel zzg) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && ffr == fudEmpty = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc (f+1)
      where
        (_,(uur,ffr)) = best uu aa zzg f 1
        zzr = tsgl ((stateEmpty,ffr),(aa, apply qq (fder ffr `cup` ll) (fhis ffr) aa))
        mm = [(b,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy)),
                 let b = if Set.null ll then a else a - aamax (bb' `mul` unit ss `red` ll), 
                 b > 0]
        (_,nn,ss,ffb,bb) = last $ sort mm
        cc = apply qq qq (fhis ffb `add` unit ss) bb
        (_,(uuc,ffc)) = best uu cc zzg f 1
        zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc),(cc, apply qq (fder ffc `cup` ll) (fhis ffc) cc))])
    best uu aa (Tree ttg) f h
      | qq /= [] && gg' /= fudEmpty && g' > g = (g', (uu'',gg'))
      | qq /= [] = (g, (uu',gg))
      | otherwise = (0,(uu,fudEmpty))
      where       
        aarr = ashuffle aa seed mult
        qq = [(good uu' aa aarr gg, (gg,uu'), xxg) | ((wmaxg,vvg,ffg),xxg) <- Map.toList ttg, 
               let (uu',ff,nn) = layerer wmaxg uu vvg ffg aa f h,
               let (a,kk) = maxd nn, a > repaRounding, let gg = depends ff (ndep lld ff kk)]
        (g, (gg,uu'), xxg) = last $ sort qq
        (g', (uu'',gg')) = best uu' aa xxg f (h+1)
    layerer wmax uu vvg ffg aa f g = parametersSystemsLayererLevelMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx' xxp' xxrr' xxrrp' f g
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xx' = applyhr uu ffg xx
        xxp' = historyRepasRed xx'   
        xxrr' = applyhr uu ffg xxrr
        xxrrp' = historyRepasRed xxrr'   
    okLevel zzg = and [wmaxg >= 0 && vvg `subset` vars aa && fvars ffg `subset` uvars uu && fund ffg `subset` vars aa |
                       (wmaxg,vvg,ffg) <- Set.toList (treesElements zzg)]
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fvars = fudsVars
    fund = fudsUnderlying
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fhis = fudsSetHistogram
    hhaa hh = historiesHistogram hh
    hshuffle hh r = fromJust $ historiesShuffle hh (fromInteger r)
    ashuffle aa seed mult = let hh = aahh aa in 
                            resize (size aa) $ foldl1 aadd [hhaa $ hshuffle hh (seed + r) | r <- [0..mult-1]]
    aamax aa = if size aa > 0 then (last $ sort $ snd $ unzip $ aall aa) else 0
    applyhr uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    aall = histogramsList
    resize z aa = fromJust $ histogramsResize z aa
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    cap = Set.intersection
    cup = Set.union
    llvv = V.fromList
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxBatchRepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  Maybe (System, DecompFud)
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxBatchRepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax fbatch mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 
    || pmax < 0 || fbatch <= 0 = Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = Nothing
  | not (qq `subset` uvars uu && vv `subset` qq) = Nothing
  | otherwise = Just $ decomp uu emptyTree 1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree && (ffr == fudEmpty || nnr == [] || ar <= repaRounding) = (uu, decompFudEmpty)
      | zz == emptyTree = decomp uur zzr (f+1)
      | (fmax > 0 && f > fmax) || mm == [] = (uu, zzdf (zztrim zz)) 
      | otherwise = decomp uuc zzc fc
      where
        (uur,ffr,nnr) = layerer uu aa f
        (ar,kkr) = maxd nnr
        ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
        zzr = tsgl ((stateEmpty,ffr'),(aa, apply qq (fder ffr') (fhis ffr') aa))
        mm = [(a,nn,ss,ff,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                 let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty,
                 (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` dom (dom (treesRoots yy))]
        (uuc,zzc,fc) = foldl zadd (uu,zz,f) (take (fromIntegral (min fbatch (fmax-f+1))) (reverse (sort mm)))
        zadd (uu,zz,f) (_,nn,ss,ffb,bb) = (uuc, zzc, f+1)
          where
            cc = apply qq qq (fhis ffb `add` unit ss) bb
            (uuc,ffc,nnc) = layerer uu cc f
            (ac,kkc) = maxd nnc
            ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
            zzc = zzadd zz (nn List.++ [((ss,ffc'),(cc, apply qq (fder ffc') (fhis ffc') cc))])
    layerer uu aa f = parametersSystemsLayererMaximumRollExcludedSelfHighestRepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
      where
        xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        z = historyRepasSize xx
        xxp = historyRepasRed xx   
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        xxrrp = historyRepasRed xxrr   
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzadd zz ll = pathsTree (treesPaths zz `add` ll)
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    depends = fudsVarsDepends
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))
    min x y = if x<y then x else y

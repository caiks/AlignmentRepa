{-# LANGUAGE RankNTypes, BangPatterns #-}

module AlignmentPracticableIORepa (
  parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u,
  parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_1,
  parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_2,
  parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_3,
  parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_4,
  parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_5,
  parametersSystemsLayererMaxRollByMExcludedSelfHighestIORepa_u,
  parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u,
  parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u_1,
  parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u_2,
  parametersSystemsLayererLevelMaxRollByMExcludedSelfHighestIORepa_u,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_1,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_2,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_3,
  parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_4,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_1,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_2,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_3,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxBatchIORepa,
  parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxIORepa,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyIORepa,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelIORepa,
  parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelIORepa,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelGoodnessIORepa,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelIORepa,
  parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelIORepa,
  parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelGoodnessIORepa
)
where
import Control.Monad
import Data.List as List
import qualified Data.Set as Set
import qualified Data.Map as Map
import qualified Data.IntMap as IntMap
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as MV
import Data.Array.Repa as R
import Data.Time
import System.Locale
import System.IO
import System.Mem
import Text.Printf
import AlignmentUtil
import Alignment
import AlignmentRandom
import AlignmentSubstrate
import AlignmentApprox
import AlignmentRepaVShape
import AlignmentRepa
import AlignmentRandomRepa
import AlignmentPracticableRepa
import GHC.Real

repaRounding :: Double 
repaRounding = 1e-6

diffTime t2 t1 = fromRational $ toRational $ diffUTCTime t2 t1 :: Double

parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    do
      printf ">>> layerer\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "layer: %d\n" l
        performGC
        t1 <- getCurrentTime
        printf ">>> tupler\n"
        printf "substrate cardinality: %d\n" $ card vv
        printf "fud cardinality: %d\n" $ card $ ffqq ff
        hFlush stdout
        let (x2,s2) = buildfftup uu vv ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality: %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else do
            printf "no tuples\n"
        performGC
        t2 <- getCurrentTime
        printf "tupler\tsearched: %d\t" $ s2
        printf "rate: %.2f\n" $ fromIntegral s2 / diffTime t2 t1
        printf "<<< tupler %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        hFlush stdout
        let (x3a,s3a) = unzip [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        let x3 = concat x3a
        let s3 = sum s3a
        if x3 /= [] then do
            printf "partitions cardinality: %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        performGC
        t3 <- getCurrentTime
        printf "parter\tsearched: %d\t" $ s3
        printf "rate: %.2f\n" $ fromIntegral s3 / diffTime t3 t2
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        hFlush stdout
        let (x4a,s4a) = unzip [roller qq | qq <- x3]
        let x4 = concat x4a
        let s4 = sum s4a
        if x4 /= [] then do
            printf "roll cardinality: %d\n" $ length x4
          else
            printf "no rolls\n"
        performGC
        t4 <- getCurrentTime
        printf "roller\tsearched: %d\t" $ s4
        printf "rate: %.2f\n" $ fromIntegral s4 / diffTime t4 t3
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        hFlush stdout
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh
        printf "fud cardinality: %d\n" $ card $ ffqq gg
        performGC
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> dervarser\n"
        hFlush stdout
        let (mm',s5) = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        performGC
        t6 <- getCurrentTime
        printf "dervarser\tsearched: %d\t" $ s5
        printf "rate: %.2f\n" $ fromIntegral s5 / diffTime t6 t5
        printf "<<< dervarser %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        hFlush stdout
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) 
          else do
            return (uu,ff,mm) 
      where
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_ui xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerRepa_ui mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_i qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (\(x1,s1) -> (List.map (\((kk,_,_),a) -> (kk,a)) x1,s1)) $
      parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_ui wmax omax uu vv ff xx xxp xxrr xxrrp
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
    card = Set.size

parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_1 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    do
      printf ">>> layerer\n"
      t1 <- getCurrentTime
      let x1 = layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
      printf "der var algn density: %.2f\n" $ (\(_,_,mm) -> maxr mm) x1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      return $ x1
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

parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_2 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_2 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    do
      printf ">>> layerer\n"
      t1 <- getCurrentTime
      x1 <- layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
      printf "der var algn density: %.2f\n" $ (\(_,_,mm) -> maxr mm) x1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      return $ x1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "layer : %d\n" l
        t1 <- getCurrentTime
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            t2 <- getCurrentTime
            printf "<<< layer %s\n" $ show $ diffUTCTime t2 t1
            layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) 
          else do
            t2 <- getCurrentTime
            printf "<<< layer %s\n" $ show $ diffUTCTime t2 t1
            return (uu,ff,mm) 
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

parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_3 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_3 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    do
      printf ">>> layerer\n"
      t1 <- getCurrentTime
      x1 <- layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      return $ x1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "layer : %d\n" l
        t1 <- getCurrentTime
        printf ">>> tuple builder\n"
        let x2 = buildfftup uu vv ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality : %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else
            printf "no tuples\n"
        t2 <- getCurrentTime
        printf "<<< tuple builder %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        let x3 = concat [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        if x3 /= [] then do
            printf "tuple partition cardinality : %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        t3 <- getCurrentTime
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        let x4 = concat [roller qq | qq <- x3]
        if x4 /= [] then do
            printf "roll cardinality : %d\n" $ length x4
          else
            printf "no rolls\n"
        t4 <- getCurrentTime
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh
        printf "fud cardinality : %d\n" $ card $ ffqq gg
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> der vars builder\n"
        let mm' = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        t6 <- getCurrentTime
        printf "<<< der vars builder %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) 
          else do
            return (uu,ff,mm) 
      where
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
    card = Set.size

parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_4 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_4 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    do
      printf ">>> layerer\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "layer: %d\n" l
        t1 <- getCurrentTime
        printf ">>> tupler\n"
        printf "substrate cardinality: %d\n" $ card vv
        printf "fud cardinality: %d\n" $ card $ ffqq ff
        hFlush stdout
        let (x2,s2) = buildfftup uu vv ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality: %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else do
            printf "no tuples\n"
        t2 <- getCurrentTime
        printf "tupler\tsearched: %d\t" $ s2
        printf "rate: %.2f\n" $ fromIntegral s2 / diffTime t2 t1
        printf "<<< tupler %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        hFlush stdout
        let (x3a,s3a) = unzip [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        let x3 = concat x3a
        let s3 = sum s3a
        if x3 /= [] then do
            printf "partitions cardinality: %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        t3 <- getCurrentTime
        printf "parter\tsearched: %d\t" $ s3
        printf "rate: %.2f\n" $ fromIntegral s3 / diffTime t3 t2
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        hFlush stdout
        let (x4a,s4a) = unzip [roller qq | qq <- x3]
        let x4 = concat x4a
        let s4 = sum s4a
        if x4 /= [] then do
            printf "roll cardinality: %d\n" $ length x4
          else
            printf "no rolls\n"
        t4 <- getCurrentTime
        printf "roller\tsearched: %d\t" $ s4
        printf "rate: %.2f\n" $ fromIntegral s4 / diffTime t4 t3
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        hFlush stdout
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh
        printf "fud cardinality: %d\n" $ card $ ffqq gg
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> dervarser\n"
        hFlush stdout
        let (mm',s5) = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        t6 <- getCurrentTime
        printf "dervarser\tsearched: %d\t" $ s5
        printf "rate: %.2f\n" $ fromIntegral s5 / diffTime t6 t5
        printf "<<< dervarser %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        hFlush stdout
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) 
          else do
            return (uu,ff,mm) 
      where
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleNoSumlayerRepa_ui xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerRepa_ui mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_i qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (\(x1,s1) -> (List.map (\((kk,_,_),a) -> (kk,a)) x1,s1)) $
      parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_ui wmax omax uu vv ff xx xxp xxrr xxrrp
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
    card = Set.size

parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_5 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u_5 
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    do
      printf ">>> layerer\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "layer: %d\n" l
        performGC
        t1 <- getCurrentTime
        printf ">>> tupler\n"
        printf "substrate cardinality: %d\n" $ card vv
        printf "fud cardinality: %d\n" $ card $ ffqq ff
        hFlush stdout
        let (x2,s2) = buildfftup uu vv ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality: %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else do
            printf "no tuples\n"
        performGC
        t2 <- getCurrentTime
        printf "tupler\tsearched: %d\t" $ s2
        printf "rate: %.2f\n" $ fromIntegral s2 / diffTime t2 t1
        printf "<<< tupler %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        hFlush stdout
        let (x3a,s3a) = unzip [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        let x3 = concat x3a
        let s3 = sum s3a
        if x3 /= [] then do
            printf "partitions cardinality: %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        performGC
        t3 <- getCurrentTime
        printf "parter\tsearched: %d\t" $ s3
        printf "rate: %.2f\n" $ fromIntegral s3 / diffTime t3 t2
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        hFlush stdout
        let (x4a,s4a) = unzip [roller qq | qq <- x3]
        let x4 = concat x4a
        let s4 = sum s4a
        if x4 /= [] then do
            printf "roll cardinality: %d\n" $ length x4
          else
            printf "no rolls\n"
        performGC
        t4 <- getCurrentTime
        printf "roller\tsearched: %d\t" $ s4
        printf "rate: %.2f\n" $ fromIntegral s4 / diffTime t4 t3
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        hFlush stdout
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh
        printf "fud cardinality: %d\n" $ card $ ffqq gg
        performGC
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> dervarser\n"
        hFlush stdout
        let (mm',s5) = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        performGC
        t6 <- getCurrentTime
        printf "dervarser\tsearched: %d\t" $ s5
        printf "rate: %.2f\n" $ fromIntegral s5 / diffTime t6 t5
        printf "<<< dervarser %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        hFlush stdout
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) 
          else do
            return (uu,ff,mm) 
      where
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleNoSumlayerRepa_ui xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerRepa_ui mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_i qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (\(x1,s1) -> (List.map (\((kk,_,_),a) -> (kk,a)) x1,s1)) $
      parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_ui wmax omax uu vv ff xx xxp xxrr xxrrp
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
    card = Set.size

parametersSystemsLayererMaxRollByMExcludedSelfHighestIORepa_u :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererMaxRollByMExcludedSelfHighestIORepa_u
  wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f = 
    do
      printf ">>> layerer\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- layer vv uu fudEmpty [] xx xxp xxrr xxrrp f 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    layer vv uu ff mm xx xxp xxrr xxrrp f l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "layer: %d\n" l
        performGC
        t1 <- getCurrentTime
        printf ">>> tupler\n"
        printf "substrate cardinality: %d\n" $ card vv
        printf "fud cardinality: %d\n" $ card $ ffqq ff
        hFlush stdout
        let (x2,s2) = buildfftup uu vv ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality: %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else do
            printf "no tuples\n"
        performGC
        t2 <- getCurrentTime
        printf "tupler\tsearched: %d\t" $ s2
        printf "rate: %.2f\n" $ fromIntegral s2 / diffTime t2 t1
        printf "<<< tupler %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        hFlush stdout
        let (x3a,s3a) = unzip [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        let x3 = concat x3a
        let s3 = sum s3a
        if x3 /= [] then do
            printf "partitions cardinality: %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        performGC
        t3 <- getCurrentTime
        printf "parter\tsearched: %d\t" $ s3
        printf "rate: %.2f\n" $ fromIntegral s3 / diffTime t3 t2
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        hFlush stdout
        let (x4a,s4a) = unzip [roller qq | qq <- x3]
        let x4 = concat x4a
        let s4 = sum s4a
        if x4 /= [] then do
            printf "roll cardinality: %d\n" $ length x4
          else
            printf "no rolls\n"
        performGC
        t4 <- getCurrentTime
        printf "roller\tsearched: %d\t" $ s4
        printf "rate: %.2f\n" $ fromIntegral s4 / diffTime t4 t3
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        hFlush stdout
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarInt f, VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh
        printf "fud cardinality: %d\n" $ card $ ffqq gg
        performGC
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> dervarser\n"
        hFlush stdout
        let (mm',s5) = buildffdervar uu' vv gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        performGC
        t6 <- getCurrentTime
        printf "dervarser\tsearched: %d\t" $ s5
        printf "rate: %.2f\n" $ fromIntegral s5 / diffTime t6 t5
        printf "<<< dervarser %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        hFlush stdout
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer vv uu' gg mm' xx' xxp' xxrr' xxrrp' f (l+1) 
          else do
            return (uu,ff,mm) 
      where
    buildfftup uu vv ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleNoSumlayerMultiEffectiveRepa_ui xmax omax bmax mmax uu vv ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerMaxRollByMRepa_ui mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_i qq
    buildffdervar uu vv ff xx xxp xxrr xxrrp = (\(x1,s1) -> (List.map (\((kk,_,_),a) -> (kk,a)) x1,s1)) $
      parametersSystemsBuilderDerivedVarsHighestNoSumlayerRepa_ui wmax omax uu vv ff xx xxp xxrr xxrrp
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
    card = Set.size


parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  IO (Maybe (System, DecompFud))
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | not (isint aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper\n"
      hFlush stdout
      t1 <- getCurrentTime
      let !hh = hhrr uu (aahh aa)
      x1 <- parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa 
        wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv hh
      t2 <- getCurrentTime
      printf "<<< decomper %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    isint = histogramsIsIntegral
    aahh aa = fromJust $ histogramsHistory aa
    hhrr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> HistoryRepa -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              let ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let wwr = fder ffr'
              let aar = apply uur ffr' aa
              let aa' = trim $ reduce uur wwr aar
              printf "derived cardinality : %d\n" $ acard $ aa'
              let zzr = tsgl ((stateEmpty,ffr'),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [(a,(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` tt]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let (a,(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `hrred` (vars aa)
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  (uuc,ffc,nnc) <- layerer uu cc f
                  printf ">>> slicing\n"
                  hFlush stdout
                  t3 <- getCurrentTime
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let wwc = fder ffc'
                  let ccc = apply uuc ffc' cc
                  let cc' = trim $ reduce uuc wwc ccc
                  printf "derived cardinality : %d\n" $ acard $ cc'
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  hFlush stdout
                  decomp uuc zzc (f+1)
    layerer uu xx f = 
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_3 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> HistoryRepa -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_3 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa =
    parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyIORepa
      wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa Set.empty

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> HistoryRepa -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_1 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1 seed
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      return $ Just $ x1
  where
    decomp uu zz f s
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa s mult f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              t3 <- getCurrentTime
              let ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let wwr = fder ffr'
              let aar = apply uur ffr' aa
              let aa' = trim $ reduce uur wwr aar
              printf "derived cardinality : %d\n" $ acard $ aa'
              let zzr = tsgl ((stateEmpty,ffr'),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              decomp uur zzr (f+1) (s + mult)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              t1 <- getCurrentTime
              let mm = V.fromList [(a,(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall bb', a > 0, ss `notin` tt]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let (a,(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `red` vv
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  (uuc,ffc,nnc) <- layerer uu cc s mult f
                  printf ">>> slicing\n"
                  t3 <- getCurrentTime
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let wwc = fder ffc'
                  let ccc = apply uuc ffc' cc
                  let cc' = trim $ reduce uuc wwc ccc
                  printf "derived cardinality : %d\n" $ acard $ cc'
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  decomp uuc zzc (f+1) (s + mult)
    layerer uu xx s mult f = 
      do
        printf ">>> repa shuffle\n"
        t1 <- getCurrentTime
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger i) | i <- [s..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    red aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_2 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> HistoryRepa -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxIORepa_2 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              t3 <- getCurrentTime
              let ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let wwr = fder ffr'
              let aar = apply uur ffr' aa
              let aa' = trim $ reduce uur wwr aar
              printf "derived cardinality : %d\n" $ acard $ aa'
              let zzr = tsgl ((stateEmpty,ffr'),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              t1 <- getCurrentTime
              let mm = V.fromList [(a,(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall bb', a > 0, ss `notin` tt]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let (a,(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `red` vv
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  (uuc,ffc,nnc) <- layerer uu cc f
                  printf ">>> slicing\n"
                  t3 <- getCurrentTime
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let wwc = fder ffc'
                  let ccc = apply uuc ffc' cc
                  let cc' = trim $ reduce uuc wwc ccc
                  printf "derived cardinality : %d\n" $ acard $ cc'
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  decomp uuc zzc (f+1)
    layerer uu xx f = 
      do
        printf ">>> repa shuffle\n"
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    red aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  IO (Maybe (System, DecompFud))
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_1 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper\n"
      t1 <- getCurrentTime
      let xx = decomp uu emptyTree 1 seed
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd xx
      t2 <- getCurrentTime
      printf "<<< decomper %s\n" $ show $ diffUTCTime t2 t1
      return $ Just $ xx 
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
    dfzz = decompFudsTreePairStateFud
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
    card = Set.size
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_2 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  IO (Maybe (System, DecompFud))
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_2 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper\n"
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1 seed
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper %s\n" $ show $ diffUTCTime t2 t1
      return $ Just $ x1
  where
    decomp uu zz f s
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa s mult f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              let ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let zzr = tsgl (stateEmpty,ffr')
              decomp uur zzr (f+1) (s + mult)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slicing\n"
              t1 <- getCurrentTime
              let mm = [(size bb,nn,ss,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                    let rrc = llsthis nn, let hhc = llfhis nn, let (_,ff) = last nn, ff /= fudEmpty,
                    ss <- qqll (cart uu (fder ff) `minus` dom (treesRoots yy)),
                    let xx = hhc `union` rrc `add` unit ss,
                    let bb = apply vv vv xx aa,
                    size bb > 0]
              printf "slices: %d\n" $ length mm
              t2 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t2 t1
              if mm == [] then
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let (z,nn,ss,bb) = last $ sort mm
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator z
                  (uuc,ffc,nnc) <- layerer uu bb s mult f
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [(ss,ffc')])
                  decomp uuc zzc (f+1) (s + mult)
    layerer uu aa s mult f = 
      do
        printf ">>> repa history\n"
        t1 <- getCurrentTime
        let !xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        t2 <- getCurrentTime
        printf "<<< repa history %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa shuffle\n"
        t1 <- getCurrentTime
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger i) | i <- [s..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let (_,ff) = last ll in if ff == fudEmpty then init ll else ll
    llsthis = Set.fromList . List.map unit . fst . unzip
    llfhis = bigcup . Set.fromList . List.map fhis . snd . unzip
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    dfzz = decompFudsTreePairStateFud
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
    card = Set.size
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_3 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  IO (Maybe (System, DecompFud))
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_3 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper\n"
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1 seed
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper %s\n" $ show $ diffUTCTime t2 t1
      return $ Just $ x1
  where
    decomp uu zz f s
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa s mult f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              let ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let zzr = tsgl (stateEmpty,ffr')
              decomp uur zzr (f+1) (s + mult)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slicing\n"
              t1 <- getCurrentTime
              let qq = ran (elem zz)
              let gg = qqff $ bigcup $ Set.map ffqq qq
              let ww = bigcup $ Set.map fder qq
              let xx = apply (vv `union` ww) gg aa              
              let mm = [(a,nn,ss,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                    let rrc = llsthis nn, let (_,ff) = last nn, ff /= fudEmpty,
                    let ww1 = fder ff, let tt = dom (treesRoots yy),
                    let xx1 = xx `mul` rrc `red` (vv `union` ww1),
                    let xx2 = xx1 `red` ww1, 
                    (ss,a) <- aall xx2, a > 0, ss `notin` tt,
                    let bb = xx1 `mul` unit ss `red` vv]
              printf "slices: %d\n" $ length mm
              t2 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t2 t1
              if mm == [] then
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let (z,nn,ss,bb) = last $ sort mm
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator z
                  (uuc,ffc,nnc) <- layerer uu bb s mult f
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [(ss,ffc')])
                  decomp uuc zzc (f+1) (s + mult)
    layerer uu aa s mult f = 
      do
        printf ">>> repa history\n"
        t1 <- getCurrentTime
        let !xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        t2 <- getCurrentTime
        printf "<<< repa history %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa shuffle\n"
        t1 <- getCurrentTime
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger i) | i <- [s..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let (_,ff) = last ll in if ff == fudEmpty then init ll else ll
    llsthis = foldl1 mul . List.map unit . fst . unzip
    zzdf zz = fromJust $ treePairStateFudsDecompFud zz
    dfzz = decompFudsTreePairStateFud
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply ww ff aa = fromJust $ setVarsFudHistogramsApply ww ff aa
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    mul = pairHistogramsMultiply
    red aa vv = setVarsHistogramsReduce vv aa
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom = relationsDomain
    ran = relationsRange
    elem = treesElements
    add qq x = Set.insert x qq
    qqll = Set.toList
    union = Set.union
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_4 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> Histogram -> 
  IO (Maybe (System, DecompFud))
parametersSystemsDecomperMaximumRollExcludedSelfHighestFmaxIORepa_4 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax < 0 || omax < 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | not (isint aa) || size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper\n"
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1 seed
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper %s\n" $ show $ diffUTCTime t2 t1
      return $ Just $ x1
  where
    decomp uu zz f s
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa s mult f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              let ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let wwr = fder ffr'
              let aar = apply (vv `union` wwr) ffr' aa
              let zzr = tsgl ((stateEmpty,ffr'),(aar, aar `red` wwr))
              decomp uur zzr (f+1) (s + mult)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slicing\n"
              t1 <- getCurrentTime
              let mm = [(a,nn,ss,bb) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall bb', a > 0, ss `notin` tt]
              printf "slices: %d\n" $ length mm
              if mm == [] then do
                  t2 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t2 t1
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let (_,nn,ss,bb) = last $ sort mm
                  let cc = bb `mul` unit ss `red` vv
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator (size cc)
                  t2 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t2 t1
                  (uuc,ffc,nnc) <- layerer uu cc s mult f
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let wwc = fder ffc'
                  let ccc = apply (vv `union` wwc) ffc' cc
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(ccc, ccc `red` wwc))])
                  decomp uuc zzc (f+1) (s + mult)
    layerer uu aa s mult f = 
      do
        printf ">>> repa history\n"
        t1 <- getCurrentTime
        let !xx = systemsHistoriesHistoryRepa_u uu $ aahh aa
        t2 <- getCurrentTime
        printf "<<< repa history %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa shuffle\n"
        t1 <- getCurrentTime
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger i) | i <- [s..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply ww ff aa = fromJust $ setVarsFudHistogramsApply ww ff aa
    aahh aa = fromJust $ histogramsHistory aa
    isint = histogramsIsIntegral
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    mul = pairHistogramsMultiply
    red aa vv = setVarsHistogramsReduce vv aa
    aall = histogramsList
    size = histogramsSize
    vars = histogramsVars
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    union = Set.union
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> HistoryRepa -> Set.Set Variable -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyIORepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa ll
  | wmax < 0 || lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa && ll `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              let ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let wwr = fder ffr'
              let aar = apply uur ffr' aa
              let aa' = trim $ reduce uur (wwr `cup` ll) aar
              printf "derived cardinality : %d\n" $ acard $ aa'
              let zzr = tsgl ((stateEmpty,ffr'),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [((b,a),(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` tt,
                    let b = fromRational a * if Set.null ll then 1 else entropy (bb' `mul` unit ss `red` ll), 
                    b > 0]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let ((b,a),(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `hrred` (vars aa)
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  printf "slice label entropy : %.2f\n" $ b
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  (uuc,ffc,nnc) <- layerer uu cc f
                  printf ">>> slicing\n"
                  hFlush stdout
                  t3 <- getCurrentTime
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let wwc = fder ffc'
                  let ccc = apply uuc ffc' cc
                  let cc' = trim $ reduce uuc (wwc `cup` ll) ccc
                  printf "derived cardinality : %d\n" $ acard $ cc'
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  hFlush stdout
                  decomp uuc zzc (f+1)
    layerer uu xx f = 
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    entropy = histogramsEntropy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    cup = Set.union
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> HistoryRepa -> Set.Set Variable -> Set.Set Variable ->
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelIORepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa ll lld
  | wmax < 0 || lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa && ll `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              let ffr' = if ar > repaRounding then depends ffr (ndep lld ffr kkr) else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let wwr = fder ffr'
              let aar = apply uur ffr' aa
              let aa' = trim $ reduce uur (wwr `cup` ll) aar
              printf "derived cardinality : %d\n" $ acard $ aa'
              let zzr = tsgl ((stateEmpty,ffr'),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [((b,a),(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` tt,
                    let b = fromRational a * if Set.null ll then 1 else entropy (bb' `mul` unit ss `red` ll), 
                    b > 0]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let ((b,a),(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `hrred` (vars aa)
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  printf "slice label entropy : %.2f\n" $ b
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  (uuc,ffc,nnc) <- layerer uu cc f
                  printf ">>> slicing\n"
                  hFlush stdout
                  t3 <- getCurrentTime
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc (ndep lld ffc kkc) else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let wwc = fder ffc'
                  let ccc = apply uuc ffc' cc
                  let cc' = trim $ reduce uuc (wwc `cup` ll) ccc
                  printf "derived cardinality : %d\n" $ acard $ cc'
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  hFlush stdout
                  decomp uuc zzc (f+1)
    layerer uu xx f = 
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fvars = fudsVars
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    entropy = histogramsEntropy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    cap = Set.intersection
    cup = Set.union
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u 
  wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx xxp xxrr xxrrp f g = 
    do
      printf ">>> layerer\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- layer uu fudEmpty [] xx xxp xxrr xxrrp 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    layer uu ff mm xx xxp xxrr xxrrp l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "level node: %d\t" g
        printf "layer: %d\n" l
        performGC
        t1 <- getCurrentTime
        printf ">>> tupler\n"
        printf "level substrate cardinality: %d\n" $ card vvg
        printf "level fud derived cardinality: %d\n" $ card (fder ffg) 
        printf "fud cardinality: %d\n" $ card $ ffqq ff
        printf "level excluded fud cardinality: %d\n" $ card $ ffqq ff `minus` ffqq ffg
        hFlush stdout
        let (x2,s2) = buildfftup uu vvg ffg ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality: %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else do
            printf "no tuples\n"
        performGC
        t2 <- getCurrentTime
        printf "tupler\tsearched: %d\t" $ s2
        printf "rate: %.2f\n" $ fromIntegral s2 / diffTime t2 t1
        printf "<<< tupler %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        hFlush stdout
        let (x3a,s3a) = unzip [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        let x3 = concat x3a
        let s3 = sum s3a
        if x3 /= [] then do
            printf "partitions cardinality: %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        performGC
        t3 <- getCurrentTime
        printf "parter\tsearched: %d\t" $ s3
        printf "rate: %.2f\n" $ fromIntegral s3 / diffTime t3 t2
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        hFlush stdout
        let (x4a,s4a) = unzip [roller qq | qq <- x3]
        let x4 = concat x4a
        let s4 = sum s4a
        if x4 /= [] then do
            printf "roll cardinality: %d\n" $ length x4
          else
            printf "no rolls\n"
        performGC
        t4 <- getCurrentTime
        printf "roller\tsearched: %d\t" $ s4
        printf "rate: %.2f\n" $ fromIntegral s4 / diffTime t4 t3
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        hFlush stdout
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarPair (VarInt f, VarInt g), VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh `funion` depends ffg (fund hh)
        printf "fud cardinality: %d\n" $ card $ ffqq gg
        performGC
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> dervarser\n"
        hFlush stdout
        let (mm',s5) = buildffdervar uu' vvg ffg gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        performGC
        t6 <- getCurrentTime
        printf "dervarser\tsearched: %d\t" $ s5
        printf "rate: %.2f\n" $ fromIntegral s5 / diffTime t6 t5
        printf "<<< dervarser %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        hFlush stdout
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer uu' gg mm' xx' xxp' xxrr' xxrrp' (l+1) 
          else do
            return (uu,ff,mm) 
      where
    buildfftup uu vvg ffg ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_ui xmax omax bmax mmax uu vvg ffg ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerRepa_ui mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_i qq
    buildffdervar uu vv ffg ff xx xxp xxrr xxrrp = (\(x1,s1) -> (List.map (\((kk,_,_),a) -> (kk,a)) x1,s1)) $
      parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_ui 
        wmax omax uu vv ffg ff xx xxp xxrr xxrrp
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    depends = fudsVarsDepends
    fder = fudsDerived
    fund = fudsUnderlying
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
    minus = Set.difference
    sgl = Set.singleton
    qqll = Set.toList
    card = Set.size

parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u_1 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u_1 
  wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx xxp xxrr xxrrp f g = 
    do
      printf ">>> layerer\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- layer uu fudEmpty [] xx xxp xxrr xxrrp 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    layer uu ff mm xx xxp xxrr xxrrp l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "level node: %d\t" g
        printf "layer: %d\n" l
        t1 <- getCurrentTime
        printf ">>> tupler\n"
        printf "level substrate cardinality: %d\n" $ card vvg
        printf "level fud derived cardinality: %d\n" $ card (fder ffg) 
        printf "fud cardinality: %d\n" $ card $ ffqq ff
        printf "level excluded fud cardinality: %d\n" $ card $ ffqq ff `minus` ffqq ffg
        hFlush stdout
        let (x2,s2) = buildfftup uu vvg ffg ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality: %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else do
            printf "no tuples\n"
        t2 <- getCurrentTime
        printf "tupler\tsearched: %d\t" $ s2
        printf "rate: %.2f\n" $ fromIntegral s2 / diffTime t2 t1
        printf "<<< tupler %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        hFlush stdout
        let (x3a,s3a) = unzip [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        let x3 = concat x3a
        let s3 = sum s3a
        if x3 /= [] then do
            printf "partitions cardinality: %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        t3 <- getCurrentTime
        printf "parter\tsearched: %d\t" $ s3
        printf "rate: %.2f\n" $ fromIntegral s3 / diffTime t3 t2
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        hFlush stdout
        let (x4a,s4a) = unzip [roller qq | qq <- x3]
        let x4 = concat x4a
        let s4 = sum s4a
        if x4 /= [] then do
            printf "roll cardinality: %d\n" $ length x4
          else
            printf "no rolls\n"
        t4 <- getCurrentTime
        printf "roller\tsearched: %d\t" $ s4
        printf "rate: %.2f\n" $ fromIntegral s4 / diffTime t4 t3
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        hFlush stdout
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarPair (VarInt f, VarInt g), VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh `funion` depends ffg (fund hh)
        printf "fud cardinality: %d\n" $ card $ ffqq gg
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> dervarser\n"
        hFlush stdout
        let (mm',s5) = buildffdervar uu' vvg ffg gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        t6 <- getCurrentTime
        printf "dervarser\tsearched: %d\t" $ s5
        printf "rate: %.2f\n" $ fromIntegral s5 / diffTime t6 t5
        printf "<<< dervarser %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        hFlush stdout
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer uu' gg mm' xx' xxp' xxrr' xxrrp' (l+1) 
          else do
            return (uu,ff,mm) 
      where
    buildfftup uu vvg ffg ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui xmax omax bmax mmax uu vvg ffg ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerRepa_ui mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_i qq
    buildffdervar uu vv ffg ff xx xxp xxrr xxrrp = (\(x1,s1) -> (List.map (\((kk,_,_),a) -> (kk,a)) x1,s1)) $
      parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_ui 
        wmax omax uu vv ffg ff xx xxp xxrr xxrrp
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    depends = fudsVarsDepends
    fder = fudsDerived
    fund = fudsUnderlying
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
    minus = Set.difference
    sgl = Set.singleton
    qqll = Set.toList
    card = Set.size

parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u_2 :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u_2 
  wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx xxp xxrr xxrrp f g = 
    do
      printf ">>> layerer\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- layer uu fudEmpty [] xx xxp xxrr xxrrp 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    layer uu ff mm xx xxp xxrr xxrrp l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "level node: %d\t" g
        printf "layer: %d\n" l
        performGC
        t1 <- getCurrentTime
        printf ">>> tupler\n"
        printf "level substrate cardinality: %d\n" $ card vvg
        printf "level fud derived cardinality: %d\n" $ card (fder ffg) 
        printf "fud cardinality: %d\n" $ card $ ffqq ff
        printf "level excluded fud cardinality: %d\n" $ card $ ffqq ff `minus` ffqq ffg
        hFlush stdout
        let (x2,s2) = buildfftup uu vvg ffg ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality: %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else do
            printf "no tuples\n"
        performGC
        t2 <- getCurrentTime
        printf "tupler\tsearched: %d\t" $ s2
        printf "rate: %.2f\n" $ fromIntegral s2 / diffTime t2 t1
        printf "<<< tupler %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        hFlush stdout
        let (x3a,s3a) = unzip [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        let x3 = concat x3a
        let s3 = sum s3a
        if x3 /= [] then do
            printf "partitions cardinality: %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        performGC
        t3 <- getCurrentTime
        printf "parter\tsearched: %d\t" $ s3
        printf "rate: %.2f\n" $ fromIntegral s3 / diffTime t3 t2
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        hFlush stdout
        let (x4a,s4a) = unzip [roller qq | qq <- x3]
        let x4 = concat x4a
        let s4 = sum s4a
        if x4 /= [] then do
            printf "roll cardinality: %d\n" $ length x4
          else
            printf "no rolls\n"
        performGC
        t4 <- getCurrentTime
        printf "roller\tsearched: %d\t" $ s4
        printf "rate: %.2f\n" $ fromIntegral s4 / diffTime t4 t3
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        hFlush stdout
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarPair (VarInt f, VarInt g), VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh `funion` depends ffg (fund hh)
        printf "fud cardinality: %d\n" $ card $ ffqq gg
        performGC
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> dervarser\n"
        hFlush stdout
        let (mm',s5) = buildffdervar uu' vvg ffg gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        performGC
        t6 <- getCurrentTime
        printf "dervarser\tsearched: %d\t" $ s5
        printf "rate: %.2f\n" $ fromIntegral s5 / diffTime t6 t5
        printf "<<< dervarser %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        hFlush stdout
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer uu' gg mm' xx' xxp' xxrr' xxrrp' (l+1) 
          else do
            return (uu,ff,mm) 
      where
    buildfftup uu vvg ffg ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleLevelNoSumlayerRepa_ui xmax omax bmax mmax uu vvg ffg ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerRepa_ui mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_i qq
    buildffdervar uu vv ffg ff xx xxp xxrr xxrrp = (\(x1,s1) -> (List.map (\((kk,_,_),a) -> (kk,a)) x1,s1)) $
      parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_ui 
        wmax omax uu vv ffg ff xx xxp xxrr xxrrp
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    depends = fudsVarsDepends
    fder = fudsDerived
    fund = fudsUnderlying
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
    minus = Set.difference
    sgl = Set.singleton
    qqll = Set.toList
    card = Set.size



parametersSystemsLayererLevelMaxRollByMExcludedSelfHighestIORepa_u :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  System -> Set.Set Variable -> Fud -> 
  HistoryRepa -> HistogramRepaRed -> HistoryRepa -> HistogramRepaRed -> Integer -> Integer ->
  IO (System, Fud, [(Set.Set Variable, Double)])
parametersSystemsLayererLevelMaxRollByMExcludedSelfHighestIORepa_u 
  wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx xxp xxrr xxrrp f g = 
    do
      printf ">>> layerer\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- layer uu fudEmpty [] xx xxp xxrr xxrrp 1
      t2 <- getCurrentTime
      printf "<<< layerer %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ x1
  where
    layer uu ff mm xx xxp xxrr xxrrp l = 
      do
        printf ">>> layer\tfud: %d\t" f
        printf "level node: %d\t" g
        printf "layer: %d\n" l
        performGC
        t1 <- getCurrentTime
        printf ">>> tupler\n"
        printf "level substrate cardinality: %d\n" $ card vvg
        printf "level fud derived cardinality: %d\n" $ card (fder ffg) 
        printf "fud cardinality: %d\n" $ card $ ffqq ff
        printf "level excluded fud cardinality: %d\n" $ card $ ffqq ff `minus` ffqq ffg
        hFlush stdout
        let (x2,s2) = buildfftup uu vvg ffg ff xx xxp xxrr xxrrp
        if x2 /= [] then do
            printf "tuple cardinality: %d\n" $ length x2
            printf "max tuple algn: %.2f\n" $ maximum $ snd $ unzip x2
          else do
            printf "no tuples\n"
        performGC
        t2 <- getCurrentTime
        printf "tupler\tsearched: %d\t" $ s2
        printf "rate: %.2f\n" $ fromIntegral s2 / diffTime t2 t1
        printf "<<< tupler %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> parter\n"
        hFlush stdout
        let (x3a,s3a) = unzip [parter uu kk bb y1 | ((kk,bb),y1) <- x2]
        let x3 = concat x3a
        let s3 = sum s3a
        if x3 /= [] then do
            printf "partitions cardinality: %d\n" $ length x3
          else
            printf "no tuple partitions\n"
        performGC
        t3 <- getCurrentTime
        printf "parter\tsearched: %d\t" $ s3
        printf "rate: %.2f\n" $ fromIntegral s3 / diffTime t3 t2
        printf "<<< parter %s\n" $ show $ diffUTCTime t3 t2
        printf ">>> roller\n"
        hFlush stdout
        let (x4a,s4a) = unzip [roller qq | qq <- x3]
        let x4 = concat x4a
        let s4 = sum s4a
        if x4 /= [] then do
            printf "roll cardinality: %d\n" $ length x4
          else
            printf "no rolls\n"
        performGC
        t4 <- getCurrentTime
        printf "roller\tsearched: %d\t" $ s4
        printf "rate: %.2f\n" $ fromIntegral s4 / diffTime t4 t3
        printf "<<< roller %s\n" $ show $ diffUTCTime t4 t3
        printf ">>> application\n"
        hFlush stdout
        let ll = [(tt,(w,ww)) | (ii,b) <- zip [ii | (yy,pp) <- x4, 
               (jj,p) <- zip (qqll yy) (V.toList pp), UV.maximum p + 1 < UV.length p,
               let ii = zip (qqll (cart uu jj)) (UV.toList p)] [1..], 
                 let w = VarPair (VarPair (VarPair (VarInt f, VarInt g), VarInt l), VarInt b), 
                 let ww = llqq $ List.map (\(_,u) -> (nnww u)) ii, 
                 let tt = trans (unit [ss `sunion` ssgl w (nnww u) | (ss,u) <- ii]) (sgl w)]
        let ll' = [(tt,(w,ww)) | (tt,(w,ww)) <- ll, 
                and [Set.size ww /= Set.size ww' || und tt /= und tt' || ttpp tt /= ttpp tt' | (tt',(w',ww')) <- ll, w > w']]
        let hh = qqff $ llqq $ fst $ unzip ll'
        let uu' = uu `uunion` (lluu $ snd $ unzip ll')
        let ffr = V.fromList $ List.map (tttr uu') $ fst $ unzip ll'
        let xx' = apply xx ffr
        let xxp' = historyRepasRed xx'
        let xxrr' = apply xxrr ffr
        let xxrrp' = historyRepasRed xxrr'
        let gg = ff `funion` hh `funion` depends ffg (fund hh)
        printf "fud cardinality: %d\n" $ card $ ffqq gg
        performGC
        t5 <- getCurrentTime
        printf "<<< application %s\n" $ show $ diffUTCTime t5 t4
        printf ">>> dervarser\n"
        hFlush stdout
        let (mm',s5) = buildffdervar uu' vvg ffg gg xx' xxp' xxrr' xxrrp'
        if mm' /= [] then do
            printf "der vars algn density: %.2f\n" $ maxr mm'
          else
            printf "no der vars sets\n"
        performGC
        t6 <- getCurrentTime
        printf "dervarser\tsearched: %d\t" $ s5
        printf "rate: %.2f\n" $ fromIntegral s5 / diffTime t6 t5
        printf "<<< dervarser %s\n" $ show $ diffUTCTime t6 t5
        printf "<<< layer %s\n" $ show $ diffUTCTime t6 t1
        hFlush stdout
        if l <= lmax && hh /= fudEmpty && (mm == [] || maxr mm' > maxr mm + repaRounding) then do
            layer uu' gg mm' xx' xxp' xxrr' xxrrp' (l+1) 
          else do
            return (uu,ff,mm) 
      where
    buildfftup uu vvg ffg ff hh hhp hhrr hhrrp = 
      parametersSystemsBuilderTupleLevelNoSumlayerMultiEffectiveRepa_ui xmax omax bmax mmax uu vvg ffg ff hh hhp hhrr hhrrp
    parter uu kk bb y1 = parametersSystemsPartitionerMaxRollByMRepa_ui mmax umax pmax uu kk bb y1
    roller qq = parametersRollerMaximumRollExcludedSelfRepa_i qq
    buildffdervar uu vv ffg ff xx xxp xxrr xxrrp = (\(x1,s1) -> (List.map (\((kk,_,_),a) -> (kk,a)) x1,s1)) $
      parametersSystemsBuilderDerivedVarsLevelHighestNoSumlayerRepa_ui 
        wmax omax uu vv ffg ff xx xxp xxrr xxrrp
    apply = historyRepasListTransformRepasApply_u
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    depends = fudsVarsDepends
    fder = fudsDerived
    fund = fudsUnderlying
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
    minus = Set.difference
    sgl = Set.singleton
    qqll = Set.toList
    card = Set.size

parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> HistoryRepa -> Tree (Integer, Set.Set Variable, Fud) -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxIORepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg =
    parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelIORepa
      lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg Set.empty Set.empty

parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> HistoryRepa -> Tree (Integer, Set.Set Variable, Fud) -> 
  Set.Set Variable -> Set.Set Variable ->
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelIORepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg ll lld
  | lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && ll `subset` vars aa) = return $ Nothing
  | not (okLevel zzg) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (uur,ffr,_) <- level uu aa zzg f 1
          if ffr == fudEmpty then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr
              let wwr = fder ffr
              let aar = apply uur ffr aa
              let aa' = trim $ reduce uur (wwr `cup` ll) aar
              printf "derived cardinality : %d\n" $ acard $ aa' `red` wwr
              let zzr = tsgl ((stateEmpty,ffr),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [((b,a),(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` tt,
                    let b = fromRational a * if Set.null ll then 1 else entropy (bb' `mul` unit ss `red` ll), 
                    b > 0]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let ((b,a),(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `hrred` (vars aa)
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  printf "slice label entropy : %.2f\n" $ b
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  (uuc,ffc,_) <- level uu cc zzg f 1
                  printf ">>> slicing\n"
                  hFlush stdout
                  t3 <- getCurrentTime
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc
                  let wwc = fder ffc
                  let ccc = apply uuc ffc cc
                  let cc' = trim $ reduce uuc (wwc `cup` ll) ccc
                  printf "derived cardinality : %d\n" $ acard $ cc' `red` wwc
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  hFlush stdout
                  decomp uuc zzc (f+1)
    level uu aa (Tree ttg) f g = foldM next (uu,fudEmpty,g) (Map.toList ttg)
      where       
        next (uu,ff,g) ((wmaxg,vvg,ffg),xxg) = 
          do
            (uuh,ffh,gh) <- level uu aa xxg f g
            (uu',gg,nn) <- layerer wmaxg uuh vvg (ffg `funion` ffh) aa f gh
            let (a,kk) = maxd nn
            let gg' = if a > repaRounding then depends gg (ndep lld gg kk) else fudEmpty
            return (uu',ff `funion` gg',gh+1)
    layerer wmax uu vvg ffg xx f g =
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xx' = apply uu ffg xx
        let !xxp' = historyRepasRed xx'   
        let !xxrr' = apply uu ffg xxrr
        let !xxrrp' = historyRepasRed xxrr'   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp'
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp'
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u 
          wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx' xxp' xxrr' xxrrp' f g
    okLevel zzg = and [wmaxg >= 0 && vvg `subset` vars aa && fvars ffg `subset` uvars uu && fund ffg `subset` vars aa |
                       (wmaxg,vvg,ffg) <- Set.toList (treesElements zzg)]
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fder = fudsDerived
    fvars = fudsVars
    fund = fudsUnderlying
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    entropy = histogramsEntropy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    cap = Set.intersection
    cup = Set.union
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelGoodnessIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> HistoryRepa -> Tree (Integer, Set.Set Variable, Fud) -> 
  Set.Set Variable -> Set.Set Variable ->
  (System -> HistoryRepa -> HistoryRepa -> Fud -> Double) -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelMinEntropyDeLabelGoodnessIORepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg ll lld good
  | lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && ll `subset` vars aa) = return $ Nothing
  | not (okLevel zzg) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (_,(uur,ffr)) <- best uu aa zzg f 1
          if ffr == fudEmpty then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr
              let wwr = fder ffr
              let aar = apply uur ffr aa
              let aa' = trim $ reduce uur (wwr `cup` ll) aar
              printf "derived cardinality : %d\n" $ acard $ aa' `red` wwr
              let zzr = tsgl ((stateEmpty,ffr),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [((b,a),(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` tt,
                    let b = fromRational a * if Set.null ll then 1 else entropy (bb' `mul` unit ss `red` ll), 
                    b > 0]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let ((b,a),(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `hrred` (vars aa)
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  printf "slice label entropy : %.2f\n" $ b
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  (_,(uuc,ffc)) <- best uu cc zzg f 1
                  printf ">>> slicing\n"
                  hFlush stdout
                  t3 <- getCurrentTime
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc
                  let wwc = fder ffc
                  let ccc = apply uuc ffc cc
                  let cc' = trim $ reduce uuc (wwc `cup` ll) ccc
                  printf "derived cardinality : %d\n" $ acard $ cc' `red` wwc
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  hFlush stdout
                  decomp uuc zzc (f+1)
    best uu aa (Tree ttg) f h 
      | ttg == Map.empty = return (0,(uu,fudEmpty))
      | otherwise = do       
        let z = historyRepasSize aa
        let aarr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u aa (fromInteger seed + i*z) | i <- [1..]]
        qq' <- forM (zip (Map.toList ttg) [(1 :: Int)..]) $ (\(((wmaxg,vvg,ffg),xxg),i) -> do 
                printf ">>> frame\tfud: %d\t" f
                printf "level node: %d\t" h
                printf "frame: %d\n" $ i
                hFlush stdout
                t1 <- getCurrentTime
                (uu',ff,nn) <- layerer wmaxg uu vvg ffg aa f h
                let (a,kk) = maxd nn
                let gg = if a > repaRounding then depends ff (ndep lld ff kk) else fudEmpty
                let g = good uu' aa aarr gg
                printf "goodness : %.2f\n" $ g
                t2 <- getCurrentTime
                printf "<<< frame %s\n" $ show $ diffUTCTime t2 t1
                return (a,(g, (gg,uu'), xxg)))
        let qq = [b | (a,b) <- qq', a > repaRounding]
        let (g, (gg,uu'), xxg) = if qq /= [] then (last $ sort qq) else (0, (fudEmpty,uu), emptyTree)
        (g', (uu'',gg')) <- best uu' aa xxg f (h+1)
        if gg' /= fudEmpty && g' > g then
            return (g', (uu'',gg'))
          else 
            return (g, (uu',gg))
    layerer wmax uu vvg ffg xx f g =
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xx' = apply uu ffg xx
        let !xxp' = historyRepasRed xx'   
        let !xxrr' = apply uu ffg xxrr
        let !xxrrp' = historyRepasRed xxrr'   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp'
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp'
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u 
          wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx' xxp' xxrr' xxrrp' f g
    okLevel zzg = and [wmaxg >= 0 && vvg `subset` vars aa && fvars ffg `subset` uvars uu && fund ffg `subset` vars aa |
                       (wmaxg,vvg,ffg) <- Set.toList (treesElements zzg)]
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fder = fudsDerived
    fvars = fudsVars
    fund = fudsUnderlying
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    entropy = histogramsEntropy
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    cap = Set.intersection
    cup = Set.union
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> HistoryRepa -> Set.Set Variable -> Set.Set Variable ->
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelIORepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax mult seed uu vv aa ll lld
  | wmax < 0 || lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && vv `subset` vars aa && ll `subset` vars aa) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              let ffr' = if ar > repaRounding then depends ffr (ndep lld ffr kkr) else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let wwr = fder ffr'
              let aar = apply uur ffr' aa
              let aa' = trim $ reduce uur (wwr `cup` ll) aar
              printf "derived cardinality : %d\n" $ acard $ aa'
              let zzr = tsgl ((stateEmpty,ffr'),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [((b,a),(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` tt,
                    let b = if Set.null ll then a else (a - aamax (bb' `mul` unit ss `red` ll)), 
                    b > 0]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let ((b,a),(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `hrred` (vars aa)
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  printf "slice label non-modal size : %d\n" $ numerator b
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  (uuc,ffc,nnc) <- layerer uu cc f
                  printf ">>> slicing\n"
                  hFlush stdout
                  t3 <- getCurrentTime
                  let (ac,kkc) = maxd nnc
                  let ffc' = if ac > repaRounding then depends ffc (ndep lld ffc kkc) else fudEmpty
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
                  let wwc = fder ffc'
                  let ccc = apply uuc ffc' cc
                  let cc' = trim $ reduce uuc (wwc `cup` ll) ccc
                  printf "derived cardinality : %d\n" $ acard $ cc'
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc'),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  hFlush stdout
                  decomp uuc zzc (f+1)
    layerer uu xx f = 
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    fvars = fudsVars
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    aamax aa = if histogramsSize aa > 0 then (last $ sort $ snd $ unzip $ aall aa) else 0
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    cap = Set.intersection
    cup = Set.union
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))


parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> HistoryRepa -> Tree (Integer, Set.Set Variable, Fud) -> 
  Set.Set Variable -> Set.Set Variable ->
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperLevelMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelIORepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg ll lld
  | lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && ll `subset` vars aa) = return $ Nothing
  | not (okLevel zzg) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (uur,ffr,_) <- level uu aa zzg f 1
          if ffr == fudEmpty then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr
              let wwr = fder ffr
              let aar = apply uur ffr aa
              let aa' = trim $ reduce uur (wwr `cup` ll) aar
              printf "derived cardinality : %d\n" $ acard $ aa' `red` wwr
              let zzr = tsgl ((stateEmpty,ffr),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [((b,a),(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` tt,
                    let b = if Set.null ll then a else (a - aamax (bb' `mul` unit ss `red` ll)), 
                    b > 0]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let ((b,a),(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `hrred` (vars aa)
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  printf "slice label non-modal size : %d\n" $ numerator b
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  (uuc,ffc,_) <- level uu cc zzg f 1
                  printf ">>> slicing\n"
                  hFlush stdout
                  t3 <- getCurrentTime
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc
                  let wwc = fder ffc
                  let ccc = apply uuc ffc cc
                  let cc' = trim $ reduce uuc (wwc `cup` ll) ccc
                  printf "derived cardinality : %d\n" $ acard $ cc' `red` wwc
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  hFlush stdout
                  decomp uuc zzc (f+1)
    level uu aa (Tree ttg) f g = foldM next (uu,fudEmpty,g) (Map.toList ttg)
      where       
        next (uu,ff,g) ((wmaxg,vvg,ffg),xxg) = 
          do
            (uuh,ffh,gh) <- level uu aa xxg f g
            (uu',gg,nn) <- layerer wmaxg uuh vvg (ffg `funion` ffh) aa f gh
            let (a,kk) = maxd nn
            let gg' = if a > repaRounding then depends gg (ndep lld gg kk) else fudEmpty
            return (uu',ff `funion` gg',gh+1)
    layerer wmax uu vvg ffg xx f g =
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xx' = apply uu ffg xx
        let !xxp' = historyRepasRed xx'   
        let !xxrr' = apply uu ffg xxrr
        let !xxrrp' = historyRepasRed xxrr'   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp'
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp'
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u 
          wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx' xxp' xxrr' xxrrp' f g
    okLevel zzg = and [wmaxg >= 0 && vvg `subset` vars aa && fvars ffg `subset` uvars uu && fund ffg `subset` vars aa |
                       (wmaxg,vvg,ffg) <- Set.toList (treesElements zzg)]
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fder = fudsDerived
    fvars = fudsVars
    fund = fudsUnderlying
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    aamax aa = if histogramsSize aa > 0 then (last $ sort $ snd $ unzip $ aall aa) else 0
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    cap = Set.intersection
    cup = Set.union
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelGoodnessIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> HistoryRepa -> Tree (Integer, Set.Set Variable, Fud) -> 
  Set.Set Variable -> Set.Set Variable ->
  (System -> HistoryRepa -> HistoryRepa -> Fud -> Double) -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxLabelModalDeLabelGoodnessIORepa 
  lmax xmax omax bmax mmax umax pmax fmax mult seed uu aa zzg ll lld good
  | lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 || pmax < 0 = 
      return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (vars aa `subset` uvars uu && ll `subset` vars aa) = return $ Nothing
  | not (okLevel zzg) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    decomp uu zz f
      | zz == emptyTree =
        do
          (_,(uur,ffr)) <- best uu aa zzg f 1
          if ffr == fudEmpty then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr
              let wwr = fder ffr
              let aar = apply uur ffr aa
              let aa' = trim $ reduce uur (wwr `cup` ll) aar
              printf "derived cardinality : %d\n" $ acard $ aa' `red` wwr
              let zzr = tsgl ((stateEmpty,ffr),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [((b,a),(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall (bb' `red` fder ff), a > 0, ss `notin` tt,
                    let b = if Set.null ll then a else (a - aamax (bb' `mul` unit ss `red` ll)), 
                    b > 0]
              printf "slices: %d\n" $ V.length mm
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let ((b,a),(nn,ss,bb)) = V.head $ vectorPairsTop 1 mm
                  let cc = select uu ss bb `hrred` (vars aa)
                  printf "decomp path length : %d\n" $ length nn
                  printf "slice size : %d\n" $ numerator a
                  printf "slice label non-modal size : %d\n" $ numerator b
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  (_,(uuc,ffc)) <- best uu cc zzg f 1
                  printf ">>> slicing\n"
                  hFlush stdout
                  t3 <- getCurrentTime
                  printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc
                  let wwc = fder ffc
                  let ccc = apply uuc ffc cc
                  let cc' = trim $ reduce uuc (wwc `cup` ll) ccc
                  printf "derived cardinality : %d\n" $ acard $ cc' `red` wwc
                  let zzc = pathsTree $ treesPaths zz `add` (nn List.++ [((ss,ffc),(ccc, cc'))])
                  t4 <- getCurrentTime
                  printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
                  hFlush stdout
                  decomp uuc zzc (f+1)
    best uu aa (Tree ttg) f h 
      | ttg == Map.empty = return (0,(uu,fudEmpty))
      | otherwise = do       
        let z = historyRepasSize aa
        let aarr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u aa (fromInteger seed + i*z) | i <- [1..]]
        qq' <- forM (zip (Map.toList ttg) [(1 :: Int)..]) $ (\(((wmaxg,vvg,ffg),xxg),i) -> do 
                printf ">>> frame\tfud: %d\t" f
                printf "level node: %d\t" h
                printf "frame: %d\n" $ i
                hFlush stdout
                t1 <- getCurrentTime
                (uu',ff,nn) <- layerer wmaxg uu vvg ffg aa f h
                let (a,kk) = maxd nn
                let gg = if a > repaRounding then depends ff (ndep lld ff kk) else fudEmpty
                let g = good uu' aa aarr gg
                printf "goodness : %.2f\n" $ g
                t2 <- getCurrentTime
                printf "<<< frame %s\n" $ show $ diffUTCTime t2 t1
                return (a,(g, (gg,uu'), xxg)))
        let qq = [b | (a,b) <- qq', a > repaRounding]
        let (g, (gg,uu'), xxg) = if qq /= [] then (last $ sort qq) else (0, (fudEmpty,uu), emptyTree)
        (g', (uu'',gg')) <- best uu' aa xxg f (h+1)
        if gg' /= fudEmpty && g' > g then
            return (g', (uu'',gg'))
          else 
            return (g, (uu',gg))
    layerer wmax uu vvg ffg xx f g =
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xx' = apply uu ffg xx
        let !xxp' = historyRepasRed xx'   
        let !xxrr' = apply uu ffg xxrr
        let !xxrrp' = historyRepasRed xxrr'   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp'
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp'
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererLevelMaximumRollExcludedSelfHighestIORepa_u 
          wmax lmax xmax omax bmax mmax umax pmax uu vvg ffg xx' xxp' xxrr' xxrrp' f g
    okLevel zzg = and [wmaxg >= 0 && vvg `subset` vars aa && fvars ffg `subset` uvars uu && fund ffg `subset` vars aa |
                       (wmaxg,vvg,ffg) <- Set.toList (treesElements zzg)]
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    ndep lld ff = Set.filter (\w -> Set.null (fvars (depends ff (Set.singleton w)) `cap` lld))
    depends = fudsVarsDepends
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    funion ff gg = qqff (ffqq ff `Set.union` ffqq gg)
    fder = fudsDerived
    fvars = fudsVars
    fund = fudsUnderlying
    fhis = fudsSetHistogram
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    aamax aa = if histogramsSize aa > 0 then (last $ sort $ snd $ unzip $ aall aa) else 0
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    mul = pairHistogramsMultiply
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    cart = systemsSetVarsSetStateCartesian_u
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    cap = Set.intersection
    cup = Set.union
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))

parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxBatchIORepa :: 
  Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
  Integer -> Integer ->
  System -> Set.Set Variable -> HistoryRepa -> 
  IO (Maybe (System, DecompFud))
parametersSystemsHistoryRepasDecomperMaximumRollExcludedSelfHighestFmaxBatchIORepa 
  wmax lmax xmax omax bmax mmax umax pmax fmax fbatch mult seed uu vv aa
  | wmax < 0 || lmax < 0 || xmax <= 0 || omax <= 0 || bmax < 0 || mmax < 1 || bmax < mmax || umax < 0 
    || pmax < 0 || fbatch <= 0 = return $ Nothing
  | size aa == 0 || mult < 1 = return $ Nothing
  | not (qq `subset` uvars uu && vv `subset` qq) = return $ Nothing
  | otherwise = 
    do
      printf ">>> decomper repa\n"
      hFlush stdout
      t1 <- getCurrentTime
      x1 <- decomp uu emptyTree 1
      printf "nodes: %d\n" $ card $ treesNodes $ dfzz $ snd x1
      t2 <- getCurrentTime
      printf "<<< decomper repa %s\n" $ show $ diffUTCTime t2 t1
      hFlush stdout
      return $ Just $ x1
  where
    qq = vars aa
    decomp uu zz f
      | zz == emptyTree =
        do
          (uur,ffr,nnr) <- layerer uu aa f
          let (ar,kkr) = maxd nnr
          if ffr == fudEmpty || nnr == [] || ar <= repaRounding then
              return $ (uu, decompFudEmpty)
            else do
              printf ">>> slicing\n"
              hFlush stdout
              t3 <- getCurrentTime
              let ffr' = if ar > repaRounding then depends ffr kkr else fudEmpty
              printf "dependent fud cardinality : %d\n" $ card $ ffqq ffr'
              let wwr = fder ffr'
              let aar = apply uur ffr' aa
              let aa' = trim $ reduce uur wwr aar
              printf "derived cardinality : %d\n" $ acard $ aa'
              let zzr = tsgl ((stateEmpty,ffr'),(aar, aa'))
              t4 <- getCurrentTime
              printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
              hFlush stdout
              decomp uur zzr (f+1)
      | otherwise = 
        do
          if fmax > 0 && f > fmax then
              return $ (uu, zzdf (zztrim zz))
            else do
              printf ">>> slice  selection\n"
              hFlush stdout
              t1 <- getCurrentTime
              let mm = V.fromList [(a,(nn,ss,bb)) | (nn,yy) <- qqll (treesPlaces zz), 
                    let ((_,ff),(bb,bb')) = last nn, ff /= fudEmpty, 
                    let tt = dom (dom (treesRoots yy)),
                    (ss,a) <- aall bb', a > 0, ss `notin` tt]
              printf "slices: %d\n" $ V.length mm
              hFlush stdout
              if V.null mm then do
                  t2 <- getCurrentTime
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  return $ (uu, zzdf (zztrim zz))
                else do
                  let mmc = V.reverse $ vectorPairsTop (fromIntegral (min fbatch (fmax-f+1))) mm
                  printf "batch: %d\n" $ V.length mmc
                  t2 <- getCurrentTime
                  hFlush stdout
                  printf "<<< slice selection %s\n" $ show $ diffUTCTime t2 t1
                  (uuc,ppc,fc) <- V.foldM zadd (uu,Set.empty,f) mmc
                  printf ">>> tree\n"
                  hFlush stdout
                  t1 <- getCurrentTime
                  let !zzc = pathsTree $ treesPaths zz `union` ppc
                  t2 <- getCurrentTime
                  printf "<<< tree %s\n" $ show $ diffUTCTime t2 t1
                  hFlush stdout
                  decomp uuc zzc fc
      where
        zadd (uu,pp,f) (a,(nn,ss,bb)) = 
          do 
            printf ">>> pre-layerer\n"
            hFlush stdout
            t1 <- getCurrentTime
            printf "decomp path length : %d\n" $ length nn
            let cc = select uu ss bb `hrred` qq
            printf "slice size : %d\n" $ numerator a
            t2 <- getCurrentTime
            printf "<<< pre-layerer %s\n" $ show $ diffUTCTime t2 t1
            hFlush stdout
            (uuc,ffc,nnc) <- layerer uu cc f
            printf ">>> slicing\n"
            hFlush stdout
            t3 <- getCurrentTime
            let (ac,kkc) = maxd nnc
            let ffc' = if ac > repaRounding then depends ffc kkc else fudEmpty
            printf "dependent fud cardinality : %d\n" $ card $ ffqq ffc'
            let wwc = fder ffc'
            let ccc = apply uuc ffc' cc
            let cc' = trim $ reduce uuc wwc ccc
            printf "derived cardinality : %d\n" $ acard $ cc'
            let ppc = pp `add` (nn List.++ [((ss,ffc'),(ccc, cc'))])
            t4 <- getCurrentTime
            printf "<<< slicing %s\n" $ show $ diffUTCTime t4 t3
            hFlush stdout
            return (uuc, ppc, f+1)
    layerer uu xx f = 
      do
        printf ">>> repa shuffle\n"
        hFlush stdout
        t1 <- getCurrentTime
        let z = historyRepasSize xx
        let !xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
        t2 <- getCurrentTime
        printf "<<< repa shuffle %s\n" $ show $ diffUTCTime t2 t1
        printf ">>> repa perimeters\n"
        hFlush stdout
        t1 <- getCurrentTime
        let !xxp = historyRepasRed xx   
        let !x2 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxp
        let !xxrrp = historyRepasRed xxrr   
        let !x3 = V.maximum $ V.map UV.maximum $ histogramRepaRedsVectorArray xxrrp
        t2 <- getCurrentTime
        printf "<<< repa perimeters %s\n" $ show $ diffUTCTime t2 t1
        hFlush stdout
        parametersSystemsLayererMaximumRollExcludedSelfHighestIORepa_u 
                                        wmax lmax xmax omax bmax mmax umax pmax uu vv xx xxp xxrr xxrrp f
    zztrim = pathsTree . Set.map lltrim . treesPaths
    lltrim ll = let ((_,ff),_) = last ll in if ff == fudEmpty then init ll else ll
    zzdf zz = fromJust $ treePairStateFudsDecompFud $ funcsTreesMap fst zz
    dfzz = decompFudsTreePairStateFud
    depends = fudsVarsDepends
    qqff = fromJust . setTransformsFud
    ffqq = fudsSetTransform
    fder = fudsDerived
    apply uu ff hh = historyRepasListTransformRepasApply hh (llvv $ List.map (tttr uu) $ qqll $ ffqq ff)
    tttr uu tt = systemsTransformsTransformRepa_u uu tt
    aahh aa = fromJust $ histogramsHistory aa
    hhhr uu hh = fromJust $ systemsHistoriesHistoryRepa uu hh
    aadd xx yy = fromJust $ pairHistogramsAdd xx yy
    select uu ss hh = historyRepasHistoryRepasHistoryRepaSelection_u (hhhr uu (aahh (unit ss))) hh
    reduce uu ww hh = fromJust $ systemsHistogramRepasHistogram uu $ setVarsHistoryRepasReduce 1 ww hh
    hrred aa vv = setVarsHistoryRepasHistoryRepaReduced vv aa
    unit = fromJust . setStatesHistogramUnit . Set.singleton 
    red aa vv = setVarsHistogramsReduce vv aa
    trim = histogramsTrim
    acard = histogramsCardinality
    aall = histogramsList
    size = historyRepasSize
    vars = Set.fromList . V.toList . historyRepasVectorVar
    uvars = systemsVars
    tsgl r = Tree $ Map.singleton r emptyTree
    maxd mm = if mm /= [] then (head $ take 1 $ reverse $ sort $ flip $ mm) else (0,empty)
    llvv = V.fromList
    bigcup :: Ord a => Set.Set (Set.Set a) -> Set.Set a
    bigcup = setSetsUnion
    dom :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a
    dom = relationsDomain
    union = Set.union
    add qq x = Set.insert x qq
    qqll = Set.toList
    empty = Set.empty
    subset = Set.isSubsetOf
    card = Set.size
    notin = Set.notMember
    flip = List.map (\(a,b) -> (b,a))
    min x y = if x<y then x else y



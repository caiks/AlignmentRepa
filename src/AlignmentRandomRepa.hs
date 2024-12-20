{-# LANGUAGE RankNTypes, BangPatterns #-}

module AlignmentRandomRepa (
  historyRepasShuffle_u,
  historyRepasShuffle_u_1,
  historyRepasShuffle_u_2,
  historyRepasShuffle_u_3,
  historyRepasShuffle_u_4,
  historyRepasShuffle_u_5,
  historyRepaRegularRandomsUniform_u,
  historyRepaRegularRandomsUniform_u_1,  
  systemsDecompFudsHistoryRepasMultiplyWithShuffle,
)
where
import Control.Monad
import Foreign.Ptr
import Foreign.C.Types
import System.IO.Unsafe
import AlignmentRepa
import Data.Int
import Data.List as List
import qualified Data.Set as Set
import qualified Data.Map as Map
import qualified Data.Vector as V
import qualified Data.Vector.Algorithms.Intro as VA
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SMV
import Data.Array.Repa as R
import GHC.Real
import AlignmentRepaVShape
import AlignmentUtil
import Alignment
import System.Random

historyRepasShuffle_u_3 :: HistoryRepa -> Int -> HistoryRepa
historyRepasShuffle_u_3 aa s = HistoryRepa vaa maa saa rbb
  where
    HistoryRepa vaa maa saa raa = aa
    Z :. (!n) :. (!z) = R.extent raa
    vv1 = UV.imap (\i a -> (i `div` z, a, i)) $ UV.fromListN (z*n) $ (drop 1 (randoms (mkStdGen s) :: [Double]))
    vv2 = UV.create $ do
      mv <- UV.unsafeThaw vv1
      VA.sort mv
      return mv
    paa = UV.map (\(_,_,i) -> i) vv2
    rbb = R.fromUnboxed ((Z :. n :. z) :: DIM2) $ UV.unsafeBackpermute (R.toUnboxed raa) paa

historyRepasShuffle_u_1 :: HistoryRepa -> Int -> HistoryRepa
historyRepasShuffle_u_1 aa s = HistoryRepa vaa maa saa rbb
  where
    HistoryRepa vaa maa saa raa = aa
    Z :. (!n) :. (!z) = R.extent raa
    vv1 = UV.imap (\i a -> (i `div` z, a, i)) $ UV.fromListN (z*n) $ (randoms (mkStdGen s) :: [Double])
    vv2 = UV.create $ do
      mv <- UV.unsafeThaw vv1
      VA.sort mv
      return mv
    paa = UV.map (\(_,_,i) -> i) vv2
    rbb = R.fromUnboxed ((Z :. n :. z) :: DIM2) $ UV.unsafeBackpermute (R.toUnboxed raa) paa

iterateUntilM :: (Monad m) => (a -> Bool) -> (a -> m a) -> a -> m a
iterateUntilM p f v 
    | p v       = return v
    | otherwise = f v >>= iterateUntilM p f

historyRepasShuffle_u_2 :: HistoryRepa -> Int -> HistoryRepa
historyRepasShuffle_u_2 aa s = HistoryRepa vaa maa saa rbb
  where
    HistoryRepa vaa maa saa raa = aa
    Z :. (!n) :. (!z) = R.extent raa
    !zd = z-1
    !qaa = R.toUnboxed raa
    !rbb = R.fromUnboxed ((Z :. n :. z) :: DIM2) $ unsafePerformIO $ do
      setStdGen (mkStdGen s)
      qbb <- MV.replicate (n*z) (-1)
      forM_ [0 .. n-1] $ (\q -> do 
        let !qz = q*z
        forM_ [0 .. z-1] $ (\i -> do 
          j <- randomRIO (0,zd) :: IO Int
          y <- MV.unsafeRead qbb (qz+j)
          (_,j) <- iterateUntilM (\(y,_) -> y == (-1)) (\(_,j) -> do
                     let j' = (j+1) `mod` z
                     y <- MV.unsafeRead qbb (qz+j')
                     return (y,j')) (y,j)
          MV.unsafeWrite qbb (qz+j) (UV.unsafeIndex qaa (qz+i))))
      UV.unsafeFreeze qbb

-- Fisher-Yates
historyRepasShuffle_u_4 :: HistoryRepa -> Int -> HistoryRepa
historyRepasShuffle_u_4 aa s = HistoryRepa vaa maa saa rbb
  where
    HistoryRepa vaa maa saa raa = aa
    Z :. (!n) :. (!z) = R.extent raa
    !rbb = R.fromUnboxed ((Z :. n :. z) :: DIM2) $ unsafePerformIO $ do
      setStdGen (mkStdGen s)
      qbb <- UV.thaw (R.toUnboxed raa)
      forM_ [0 .. n-1] $ (\q -> do 
        let !qz = q*z
        forM_ (reverse [0 .. z-1]) $ (\i -> do 
          j <- randomRIO (0,i) :: IO Int
          x <- MV.unsafeRead qbb (qz+i)
          y <- MV.unsafeRead qbb (qz+j)
          MV.unsafeWrite qbb (qz+j) x
          MV.unsafeWrite qbb (qz+i) y))
      UV.unsafeFreeze qbb

foreign import ccall unsafe "historyShuffle_u_5" historyShuffle_u_5
    :: CLLong -> CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> IO ()

historyRepasShuffle_u_5 :: HistoryRepa -> Int -> HistoryRepa
historyRepasShuffle_u_5 aa s = HistoryRepa vvaa maa saa rbb
  where
    HistoryRepa vvaa maa saa raa = aa
    Z :. (!n) :. (!z) = R.extent raa
    !vaa = SV.unsafeCast (UV.convert (R.toUnboxed raa)) :: SV.Vector CShort
    !rbb = R.fromUnboxed ((Z :. n :. z) :: DIM2) $ SV.convert $ SV.unsafeCast $ unsafePerformIO $ do
      vbb <- SMV.replicate (n*z) (-1)
      SV.unsafeWith vaa $ \paa -> do
        SMV.unsafeWith vbb $ \pbb -> do
          historyShuffle_u_5 (fromIntegral n) (fromIntegral z) paa (fromIntegral s) pbb
      SV.unsafeFreeze vbb 

foreign import ccall unsafe "historyShuffle_u" historyShuffle_u
    :: CLLong -> CLLong -> CLLong -> Ptr CShort -> IO ()

historyRepasShuffle_u :: HistoryRepa -> Int -> HistoryRepa
historyRepasShuffle_u aa s = HistoryRepa vvaa maa saa rbb
  where
    HistoryRepa vvaa maa saa raa = aa
    Z :. (!n) :. (!z) = R.extent raa
    !vbb = SV.unsafeCast (UV.convert (R.toUnboxed raa)) :: SV.Vector CShort
    !rbb = R.fromUnboxed ((Z :. n :. z) :: DIM2) $ SV.convert $ SV.unsafeCast $ unsafePerformIO $ do
      mvbb <- SV.unsafeThaw vbb
      SMV.unsafeWith mvbb $ \pbb -> do
        historyShuffle_u (fromIntegral n) (fromIntegral z) (fromIntegral s) pbb
      SV.unsafeFreeze mvbb 

historyRepaRegularRandomsUniform_u :: Int16 -> Int -> Int -> Int -> HistoryRepa
historyRepaRegularRandomsUniform_u d n z s = 
    arraysHistoryRepaCardinal_u (UV.replicate n (fromIntegral d)) $ R.fromUnboxed (R.Z R.:. n R.:. z) $ UV.fromListN (z*n) $ (drop 1 (randomRs (0,d-1) (mkStdGen s) :: [Int16] ))

historyRepaRegularRandomsUniform_u_1 :: Int16 -> Int -> Int -> Int -> HistoryRepa
historyRepaRegularRandomsUniform_u_1 d n z s = 
    arraysHistoryRepaCardinal_u (UV.replicate n (fromIntegral d)) $ R.fromUnboxed (R.Z R.:. n R.:. z) $ UV.fromListN (z*n) $ (randomRs (0,d-1) (mkStdGen s) :: [Int16] )


systemsDecompFudsHistoryRepasMultiplyWithShuffle :: 
  Integer -> Integer -> System -> DecompFud -> HistoryRepa -> Tree ((State,Fud),(HistoryRepa,HistoryRepa))
systemsDecompFudsHistoryRepasMultiplyWithShuffle mult seed uu df aa = apply (dfzz df) (vars aa) aa
  where
    apply :: Tree (State,Fud) -> Set.Set Variable -> HistoryRepa -> Tree ((State,Fud),(HistoryRepa,HistoryRepa))
    apply zz vv aa = Tree $ llmm $ [(((ss,ff), (bb, bbxx)), apply yy vv bb) | 
      ((ss,ff),yy) <- zzll zz, let aa' = select uu ss aa, let aaxx' = shuffle aa', let ww = fder ff, 
      let bb = if size aa' > 0 then (applyFud uu ff aa' `red` (vv `cup` ww)) else empty, 
      let bbxx = if size aaxx' > 0 then (applyFud uu ff aaxx' `red` (vv `cup` ww)) else empty]
    shuffle xx 
      | z == 0 = empty
      | otherwise = xxrr
      where
        z = historyRepasSize xx
        xxrr = vectorHistoryRepasConcat_u $ V.fromListN (fromInteger mult) $ 
                 [historyRepasShuffle_u xx (fromInteger seed + i*z) | i <- [1..]]
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


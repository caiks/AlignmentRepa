{-# LANGUAGE BangPatterns, TypeSynonymInstances, FlexibleInstances, RankNTypes #-}

module AlignmentRepaVShape (
  VShape(..),
  incIndex,
  incIndex_1,
  incIndex_2,
  incIndexM_,
  toIndexM,
  toIndexPermM,
  toIndexPermM_1,
  toIndexInsertM,
  toIndexPermOffsetM
)
where
import Control.Monad
import Control.Monad.Primitive
import Control.Monad.ST
import Data.Int
import Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as MV
import Data.Array.Repa.Shape

type VShape = Vector Int

instance Shape VShape where
    {-# INLINE [1] rank #-}
    rank vv = UV.length vv

    {-# INLINE [1] zeroDim #-}
    zeroDim = UV.singleton 0

    {-# INLINE [1] unitDim #-}
    unitDim = UV.singleton 1

    {-# INLINE [1] intersectDim #-}
    intersectDim vv1 vv2 = UV.map (\(i1,i2) -> min i1 i2) $ UV.zip vv1 vv2

    {-# INLINE [1] addDim #-}
    addDim vv1 vv2 = UV.map (\(i1,i2) -> i1 + i2) $ UV.zip vv1 vv2

    {-# INLINE [1] size #-}
    size vv = UV.foldl' (*) 1 vv

    {-# INLINE [1] sizeIsValid #-}
    sizeIsValid sh = size sh > 0 && size sh <= maxBound

    {-# INLINE [1] toIndex #-}
    toIndex !vv1 !vv2 = UV.foldl' (\a (d,i) -> if a /= 0 then d*a + i else i) 0 $ UV.zip vv1 vv2

    {-# INLINE [1] fromIndex #-}
    fromIndex !vv !n = snd $ UV.unzip $ UV.postscanr' (\d (a,_) -> if a >= d then a `quotRem` d else (0, a)) (n,0) vv
--    snd $ UV.foldr' (\d (a,xx) -> let !(q,r) = a `quotRem` d in (q, r `UV.cons` xx)) (n,UV.empty) vv

    {-# INLINE [1] inShapeRange #-}
    inShapeRange vv1 vv2 vv3 = UV.all (\(i1,i2,i3) -> i3 >= i1 && i3 <= i2) $ UV.zip3 vv1 vv2 vv3

    {-# NOINLINE listOfShape #-}
    listOfShape vv = UV.toList vv

    {-# NOINLINE shapeOfList #-}
    shapeOfList ll = UV.fromList ll

    {-# INLINE deepSeq #-}
    deepSeq sh x = sh `seq` x

{-# INLINE [1] incIndex #-}
incIndex :: VShape -> VShape -> VShape
incIndex !svv !ivv = UV.create $ 
  do 
    mv <- MV.new n
    carry' mv (n-1)
  where
    !n = UV.length svv
    carry' !mv !i = 
        carry i
      where
        carry !i = do
          !x <- UV.unsafeIndexM ivv i
          let !y = x + 1
          !z <- UV.unsafeIndexM svv i
          if y == z 
            then do {MV.unsafeWrite mv i 0; if i>0 then carry (i-1) else return mv;} 
            else do {MV.unsafeWrite mv i y; if i>0 then copy (i-1) else return mv;} 
        copy !i = do {!x <- UV.unsafeIndexM ivv i; MV.unsafeWrite mv i x; if i>0 then copy (i-1) else return mv;} 

{-# INLINE [1] incIndex_1 #-}
incIndex_1 :: VShape -> VShape -> VShape
incIndex_1 !svv !ivv = snd $ UV.unzip $ 
  UV.postscanr' (\(s,i) (c,_) -> let j = i+c in if j == s then (1,0) else (0,j)) (1,0) $ UV.zip svv ivv

{-# INLINE [1] incIndex_2 #-}
incIndex_2 :: VShape -> VShape -> VShape
incIndex_2 !svv !ivv = UV.create $ do {mv <- MV.new n; loop (mv, n-1, 1)}
  where
    !n = UV.length svv
    loop (mv, i, c) = do
      let j = ivv UV.! i + c
      if j == svv UV.! i 
        then do {MV.unsafeWrite mv i 0; if i>0 then loop (mv, i-1, 1) else return mv;} 
        else do {MV.unsafeWrite mv i j; if i>0 then loop (mv, i-1, 0) else return mv;} 

{-# INLINE [1] incIndex_3 #-}
incIndex_3 :: VShape -> VShape -> VShape
incIndex_3 !svv !ivv = UV.create $ 
  do 
    mv <- MV.new n
    carry (mv,n-1)
  where
    !n = UV.length svv
    carry (mv,i) = do
      let y = ivv UV.! i + 1
      if y == svv UV.! i 
        then do {MV.unsafeWrite mv i 0; if i>0 then carry (mv,i-1) else return mv;} 
        else do {MV.unsafeWrite mv i y; if i>0 then copy (mv,i-1) else return mv;} 
    copy (mv,i) = do {MV.unsafeWrite mv i (ivv UV.! i); if i>0 then copy (mv,i-1) else return mv;} 

{-# INLINE incIndexM_ #-}
incIndexM_ :: (PrimMonad m) => VShape -> (MV.MVector (PrimState m) Int) -> m ()
incIndexM_ !svv !ivv =  
  do 
    carry (n-1)
  where
    !n = UV.length svv
    carry !i = do
      !x <- MV.unsafeRead ivv i
      let !y = x + 1
      !z <- UV.unsafeIndexM svv i
      if y == z 
        then do 
          MV.unsafeWrite ivv i 0
          if i>0 then carry (i-1) else return ()
        else do 
          MV.unsafeWrite ivv i y
          return ()

{-# INLINE toIndexM #-}
toIndexM :: (PrimMonad m) => VShape -> (MV.MVector (PrimState m) Int) -> m Int
toIndexM !svv !ivv =  
  do 
    !a0 <- MV.unsafeRead ivv 0
    accum a0 1
  where
    !n = UV.length svv
    accum !a !i = do
      if i/=n 
        then do
          !d <- UV.unsafeIndexM svv i
          !y <- MV.unsafeRead ivv i
          accum (d*a + y) (i+1)
        else do
          return a 

toIndexPermM_1 :: (PrimMonad m) => VShape -> VShape -> (MV.MVector (PrimState m) Int) -> m Int
toIndexPermM_1 !skk !pkk !ivv = UV.foldM' (\a (d,j) -> do !y <- MV.unsafeRead ivv j; return (d*a + y)) 0 $ UV.zip skk pkk 

{-# INLINE toIndexPermM #-}
toIndexPermM :: (PrimMonad m) => VShape -> VShape -> (MV.MVector (PrimState m) Int) -> m Int
toIndexPermM !skk !pkk !ivv =  
  do 
    accum 0 0
  where
    !n = UV.length skk
    accum !a !i = do
      if i/=n 
        then do
          let !j = UV.unsafeIndex pkk i
          !y <- MV.unsafeRead ivv j
          if a/=0 
            then do
              let !d = UV.unsafeIndex skk i
              accum (d*a + y) (i+1)
            else do
              accum y (i+1)
        else do
          return a

{-# INLINE toIndexPermM_2 #-}
toIndexPermM_2 :: (PrimMonad m) => VShape -> VShape -> (MV.MVector (PrimState m) Int) -> m Int
toIndexPermM_2 !skk !pkk !ivv =  
  do 
    accum 0 0
  where
    !n = UV.length skk
    accum !a !i = do
      if i/=n 
        then do
          !j <- UV.unsafeIndexM pkk i
          !y <- MV.unsafeRead ivv j
          if a/=0 
            then do
              !d <- UV.unsafeIndexM skk i
              accum (d*a + y) (i+1)
            else do
              accum y (i+1)
        else do
          return a

{-# INLINE toIndexInsertM #-}
toIndexInsertM :: (PrimMonad m) => Int -> Int -> Int -> VShape -> (MV.MVector (PrimState m) Int) -> m Int
toIndexInsertM !u !r !q !svv !ivv =  
  do
    if u /= 0
      then do 
        !a0 <- MV.unsafeRead ivv 0
        accumins a0 1
      else do
        accum q 0
  where
    !n = UV.length svv
    accumins !a !i = do
      if i /= n 
        then do
          if u /= i
            then do 
              !d <- UV.unsafeIndexM svv i
              !y <- MV.unsafeRead ivv i
              accumins (d*a + y) (i+1)
            else do
              accum (r*a + q) i
        else do
          if u /= i
            then do 
              return a 
            else do
              return (r*a + q)
    accum !a !i = do
      if i /= n 
        then do
          !d <- UV.unsafeIndexM svv i
          !y <- MV.unsafeRead ivv i
          accum (d*a + y) (i+1)
        else do
          return a 

{-# INLINE toIndexPermOffsetM #-}
toIndexPermOffsetM :: (PrimMonad m) => VShape -> VShape -> Int -> (MV.MVector (PrimState m) Int16) -> m Int
toIndexPermOffsetM !skk !pkk !r !ivv =  
  do 
    accum 0 0
  where
    !n = UV.length skk
    accum !a !i = do
      if i/=n 
        then do
          let !j = UV.unsafeIndex pkk i
          !y <- MV.unsafeRead ivv (r+j)
          if a/=0 
            then do
              let !d = UV.unsafeIndex skk i
              accum (d*a + (fromIntegral y)) (i+1)
            else do
              accum (fromIntegral y) (i+1)
        else do
          return a



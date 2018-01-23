{-# LANGUAGE OverloadedStrings, DeriveGeneric, RankNTypes #-}

module AlignmentAesonRepa (
  persistentsHistoryRepa,
  systemsHistoryRepasPersistent,
  systemsHistoryRepasPersistent_u
)
where
import Data.List as List
import qualified Data.Set as Set
import qualified Data.Map as Map
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Data.Array.Repa as R
import Control.Applicative
import Control.Monad
import Data.Aeson hiding (Value)
import Data.Monoid
import GHC.Real
import GHC.Generics
import AlignmentUtil
import Alignment hiding (derived)
import AlignmentAeson
import AlignmentRepaVShape
import AlignmentRepa

systemsListVariablesPersistent_u :: System -> [Variable] -> SystemPersistent
systemsListVariablesPersistent_u uu ll = 
    SystemPersistent [VariablePersistent {var = vars v, values = map vals (qqll (uu `uat` v))} | v <- ll]
  where
    uat uu v = fromJust $ systemsVarsValues uu v
    vars (VarPair (v,w)) = "<" ++ vars v ++ "," ++ vars w ++ ">"
    vars (VarInt i) = show i
    vars (VarStr s) = s
    vals (ValInt i) = show i
    vals (ValStr s) = s
    qqll = Set.toList

persistentsHistoryRepa :: HistoryPersistent -> Maybe (System, HistoryRepa)
persistentsHistoryRepa hh   
  | uu' == Nothing = Nothing
  | hr' == Nothing = Nothing
  | otherwise = Just $ (uu,hr)
  where
    uu' = persistentsSystem $ hsystem hh
    uu = fromJust uu'
    SystemPersistent ll' = hsystem hh
    ll = [svar (var x) | x <- ll']
    hr' = systemsListVariablesListsListsHistoryRepa uu ll (hstates hh)
    hr = fromJust hr' 
    svar = stringsVariable

systemsHistoryRepasPersistent :: System -> HistoryRepa -> Maybe HistoryPersistent
systemsHistoryRepasPersistent uu hh 
  | not (vars hh `subset` uvars uu) = Nothing
  | otherwise = Just $ systemsHistoryRepasPersistent_u uu hh
  where
    vars = Set.fromList . V.toList . historyRepasVectorVar
    uvars = systemsVars
    subset = Set.isSubsetOf

systemsHistoryRepasPersistent_u :: System -> HistoryRepa -> HistoryPersistent
systemsHistoryRepasPersistent_u uu hh = 
    HistoryPersistent {hsystem = uupp uu (vars hh), hstates = historyRepasListsList hh}
  where
    uupp = systemsListVariablesPersistent_u
    vars = V.toList . historyRepasVectorVar


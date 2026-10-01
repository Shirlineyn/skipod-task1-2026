#!/usr/bin/env wolframscript
(* Эталон для варианта 12: исходная задача Дирихле в D методом конечных
   элементов, без фиктивных областей. Печатает интеграл u по D и значения
   в контрольных точках; tools/compare_reference.py сверяет с ними сеточные
   решения. *)
reg = ImplicitRegion[x^2 - 4 y^2 > 1 && 1 < x < 3, {x, y}];
pts = {{2., 0.}, {1.5, 0.3}, {2.5, -0.8}, {2.8, 1.}, {1.2, -0.2}};
Do[
  u = NDSolveValue[{-Laplacian[w[x, y], {x, y}] == 1,
       DirichletCondition[w[x, y] == Sin[x + y], True]},
      w, {x, y} \[Element] reg,
      Method -> {"FiniteElement", "MeshOptions" -> {"MaxCellMeasure" -> mcm, "MeshOrder" -> 2}}];
  int = NIntegrate[u[x, y], {x, y} \[Element] u["ElementMesh"]];
  Print["mcm=", mcm, " int=", ToString[NumberForm[int, 12]], " vals=",
        ToString[NumberForm[u @@@ pts, 12]]],
  {mcm, {0.001, 0.0002, 0.00005}}];
Print["points=", pts];

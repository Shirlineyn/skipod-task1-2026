(* Symbolic and exact-arithmetic checks used in the report. *)
checks = <||>;
G[y_] := y Sqrt[1+4 y^2]/2 + ArcSinh[2 y]/4;
checks["antiderivative"] = FullSimplify[D[G[y],y] == Sqrt[1+4 y^2], Element[y,Reals]];
area = Integrate[3-Sqrt[1+4 y^2],{y,-Sqrt[2],Sqrt[2]}];
checks["area"] = FullSimplify[area == 3 Sqrt[2]-ArcSinh[2 Sqrt[2]]/2];
a = {{a11,a12},{a12,a22}}; d={d1,d2}; b={b1,b2}; r=b-a.d;
checks["functional"] = Expand[d.a.d/2-b.d+(b+r).d/2] == 0;
parts[m_,p_] := Module[{lens,offs,ends},
  lens=Table[Quotient[m-1,p]+Boole[c<Mod[m-1,p]],{c,0,p-1}];
  offs=Most[FoldList[Plus,0,lens]];
  ends=offs+lens; ends[[-1]]=m;
  {lens,offs,ends,ends-offs+1}];
checks["domain_partition"] = And@@Flatten[Table[
  Module[{v=parts[m,p]},
    Total[v[[1]]]==m-1 && v[[2,1]]==0 && v[[3,-1]]==m &&
    Rest[v[[2]]]==Most[v[[3]]] && Max[v[[4]]]-Min[v[[4]]]<=1 &&
    Min[v[[4]]]==Floor[m/p]+1 && Max[v[[4]]]==Ceiling[m/p]+1],
  {m,2,200},{p,1,m-1}]];
checks["80x80_P2"] = parts[80,2][[4]] == {41,41} && 81/41<2;
checks["owned_ratio_impossible"] = 79/39>2;
Print["checks=",checks];
Print["area=",N[area,16]];
Print["domain_ratio=",81/41,"; owned_worst_ratio=",79/39];
If[!And@@Values[checks],Exit[1]];

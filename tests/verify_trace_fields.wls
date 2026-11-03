(* Independent check of the reported whole-field norms using Mathematica. *)
work=Environment["SKIPOD_REPORT_WORK"];
root=DirectoryName[DirectoryName[$InputFileName]];
rows=Import[FileNameJoin[{root,"report","data","summary.json"}],"RawJSON"];
read[path_] := Module[{s,mn,box,w},
 s=OpenRead[path,BinaryFormat->True];
 mn=BinaryReadList[s,"Integer32",2]; box=BinaryReadList[s,"Real64",4];
 w=BinaryReadList[s,"Real64"]; Close[s]; {mn,box,w}];
ok=True;
Do[
 key=row["grid"];
 a=read[FileNameJoin[{work,"solution_"<>key<>".bin"}]];
 b=read[FileNameJoin[{work,"reference_"<>key<>"_13.bin"}]];
 cell=(a[[2,2]]-a[[2,1]])/a[[1,1]]*(a[[2,4]]-a[[2,3]])/a[[1,2]];
 err=a[[3]]-b[[3]];
 einf=Max[Abs[err]]; e2=Sqrt[cell Total[err^2]];
 good=a[[1]]==b[[1]] && Abs[einf-row["iteration_error_inf"]]<10^-12 &&
      Abs[e2-row["iteration_error_l2"]]<10^-12;
 Print[key," einf=",InputForm[einf]," e2=",InputForm[e2]," match=",good];
 ok=ok&&good,
 {row,rows}];
If[!ok,Exit[1]];

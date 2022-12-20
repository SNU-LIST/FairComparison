function MDH1=readfirstmdh(fileid)
% readfirst MDH info. for SIEMENS vb17
% yhnam83@gmail.com

fid=fopen(fileid,'r');
global_header.size=fread(fid,1,'int32');
global_header.content=fread(fid,global_header.size-4,'uint8');
global_header.content=char(global_header.content);

MDH1.ulFlagsAndDMALength=fread(fid,1,'uint32');
MDH1.lMeasUID=fread(fid,1,'int32');
MDH1.ulScanCounter=fread(fid,1,'uint32');
MDH1.ulTimeStamp=fread(fid,1,'uint32');
MDH1.ulPMUTimeStamp=fread(fid,1,'uint32');
MDH1.aulEvalInfoMask1=fread(fid,1,'uint32');
MDH1.aulEvalInfoMask2=fread(fid,1,'uint32');
MDH1.ushSamplesInScan=fread(fid,1,'uint16');
MDH1.ushUsedChannels=fread(fid,1,'uint16');
% 4+4+4+4+4+4+4+2+2=32 bytes.

MDH1.ushLine=fread(fid,1,'uint16');
MDH1.ushAcquisition=fread(fid,1,'uint16');
MDH1.ushSlice=fread(fid,1,'uint16');
MDH1.ushPartition=fread(fid,1,'uint16');
MDH1.ushEcho=fread(fid,1,'uint16');
MDH1.ushPhase=fread(fid,1,'uint16');
MDH1.ushRepetition=fread(fid,1,'uint16');
MDH1.ushSet=fread(fid,1,'uint16');
MDH1.ushSeg=fread(fid,1,'uint16');
MDH1.ushIda=fread(fid,1,'uint16');
MDH1.ushIdb=fread(fid,1,'uint16');
MDH1.ushIdc=fread(fid,1,'uint16');
MDH1.ushIdd=fread(fid,1,'uint16');
MDH1.ushIde=fread(fid,1,'uint16');
% 2*14=28 bytes.

MDH1.ushPre=fread(fid,1,'uint16');
MDH1.ushPost=fread(fid,1,'uint16');
% 2*2=4 bytes.

MDH1.ushKSpaceCentreColumn=fread(fid,1,'uint16');
MDH1.ushCoilSelect=fread(fid,1,'uint16');
MDH1.fReadOutOffcentre=fread(fid,1,'float');
MDH1.ulTimeSinceLastRF=fread(fid,1,'uint32');
MDH1.ushKSpaceCentreLineNo=fread(fid,1,'uint16');
MDH1.ushKSpaceCentrePartitionNo=fread(fid,1,'uint16');
MDH1.aushIceProgramPara1=fread(fid,1,'uint16');
MDH1.aushIceProgramPara2=fread(fid,1,'uint16');
MDH1.aushIceProgramPara3=fread(fid,1,'uint16');
MDH1.aushIceProgramPara4=fread(fid,1,'uint16');
MDH1.aushFreePara1=fread(fid,1,'uint16');
MDH1.aushFreePara2=fread(fid,1,'uint16');
MDH1.aushFreePara3=fread(fid,1,'uint16');
MDH1.aushFreePara4=fread(fid,1,'uint16');
% 2*12+4*2=32 bytes;

MDH1.sSlicePosVec_flSag=fread(fid,1,'float');
MDH1.sSlicePosVec_flCor=fread(fid,1,'float');
MDH1.sSlicePosVec_flTra=fread(fid,1,'float');
MDH1.aflQuaternion1=fread(fid,1,'float');
MDH1.aflQuaternion2=fread(fid,1,'float');
MDH1.aflQuaternion3=fread(fid,1,'float');
MDH1.aflQuaternion4=fread(fid,1,'float');
% 4*7=28 bytes;

MDH1.ushChannelId=fread(fid,1,'uint16');
MDH1.ushPTABPosNeg=fread(fid,1,'uint16');
% 2*2=4 bytes;
fclose(fid);


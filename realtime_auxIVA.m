%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%It is written by Yongbao for Professor Zhou's students in 2018-11-09 
%edited by Xiuxiang and Yongbao to process aux-iva in 2019-12-05                   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y_out_all,fs,energy_iva,nb_frames]=realtime_auxIVA(path,inputname)
%-----------------------------About sigal process in frame-----------------
frame_size   = 1024;
FFT_size     = 2*frame_size;
FFT_bins     = frame_size+1;
epsi = 1e-6;
%inputname='2018_nova3_0_90';
%path='../wav/';
read_pcm     =0; %1:read pcm 0:read wav
if(read_pcm==1)
    fileId = fopen([path,inputname,'.pcm'],'r');
    x = fread(fileId,inf,'int16');
    fclose(fileId);
    Channel      =1;
    for i=1:Channel
        x_in(:,i)=x(i:Channel:end);
    end
else
    [x_in,fs]=audioread([path,inputname],'native') ;
end
x=double(int16(x_in));
N=size(x,1);
channel=size(x,2);
nb_frames       = fix((size(x,1)-frame_size)/frame_size);
x_old           = zeros(frame_size,channel);
y_frame_old     = zeros(frame_size,channel);
win             = sqrt(hanning(FFT_size,'periodic'));
X_frame         = zeros(FFT_bins,channel);
Gain            = ones(FFT_bins,channel);
bins_70hz       = floor(70/(fs/FFT_size));%
Y_frame_out=zeros(FFT_bins,channel);
Wp=zeros(channel,channel,FFT_bins);%%%
for k=1:FFT_bins
    Wp(:,:,k)=eye(channel);
end
iter_num=1;
e=eye(channel);
%%
Vtmp=zeros(channel,channel,nb_frames);
V_smooth_old1=zeros(channel,channel,FFT_bins);
V_smooth_old2=zeros(channel,channel,FFT_bins);
V_smooth=zeros(channel,channel,FFT_bins);
aifa=0.95;
r=zeros(1,channel);%%%
conX=zeros(channel,channel,FFT_bins);
energy_iva=zeros(nb_frames,channel);%%
%----------------------------frame loop start------------------------------
for frame_idx=1:nb_frames
    fprintf('%d / %d \n',frame_idx,nb_frames);%%
    pos      = frame_size * (frame_idx-1) + 1;
    % 50%overlap and window
    x_new          = x(pos:pos+frame_size-1,:);
    x_frame        = [x_old;x_new];
    x_old          = x_new;
    % FFT
    win_copy       = repmat(win,1,size(x_frame,2));
    tmp1           = x_frame.*win_copy;
    temp           = fft(tmp1)./FFT_size;
    X_frame        = temp(1:FFT_bins,:);
    
    %Do process
    Gain(1:bins_70hz)=0.01;%
    X_frame = X_frame.*Gain;
    for k=1:FFT_bins
        Y_frame_out(k,:) = Wp(:,:,k)*X_frame(k,:).';%%%%%
    end

    Power_frame = max(abs(Y_frame_out).^2,eps).';%%%%
    sumP2_freq = squeeze(sum(Power_frame,2));
    coffs = sumP2_freq .^(-1); %%% G'/r %%for Aux_iva; coffs = 2 * sumP2_freq .^(-2/3) / 3;
    coffs = 2 * sumP2_freq .^(-2/3) / 3;%%%%
    %%
    for k=1:FFT_bins
        conX(:,:,k)=X_frame(k,:).'*conj(X_frame(k,:));%%%%
    end
    %         if flag == 1
    for iter=1:iter_num
        for ch=1:channel
            coff=sum(abs(Y_frame_out(:,ch)).^2);
            r(:,ch)=sqrt(coff);
            Gr=1./(r(:,ch)+epsi);
%             Gr=coffs(ch);
            for k=1:FFT_bins
                V_new=Gr.*conX(:,:,k);% 
                if(ch==1)
                    V_smooth(:,:,k)=aifa.*V_smooth_old1(:,:,k)+(1-aifa).*V_new;
                else
                    V_smooth(:,:,k)=aifa.*V_smooth_old2(:,:,k)+(1-aifa).*V_new;
                end
                if(ch==1)
                    V_smooth_old1(:,:,k)=V_smooth(:,:,k);
                else
                    V_smooth_old2(:,:,k)=V_smooth(:,:,k);
                end
                if frame_idx==1
                    %                     V_smooth(:,:,k)=V_new;
                    V_smooth(:,:,k)=eye(2);
                end
                if(frame_idx>=2)%%%
                    %%%%Update of demixing matrix
                    Vtmp=V_smooth(:,:,k);%%%%
                    Wtmp = pinv(Wp(:,:,k)*Vtmp+1e-5*eye(ch))*e(:,ch);
                    Wtmp = Wtmp / sqrt(Wtmp' * Vtmp * Wtmp);
                    Wp(ch,:,k) = Wtmp';%%%
                end
            end
        end%%%
        %%%
        %     else
        %         Wp(:,:,k)= Wp(:,:,k-1);
        %     end
        for k=1:FFT_bins
            Wp(:,:,k)=diag(diag(pinv(Wp(:,:,k))))*Wp(:,:,k);
            Y_frame_out(k,:) = Wp(:,:,k)*X_frame(k,:).';
        end
                
    end
%     %IFFT
    temp=[Y_frame_out;flipud(conj(Y_frame_out(2:FFT_bins-1,1:channel)))];
    y_frame=real(ifft(temp.*FFT_size));
    %OLA
    y_frame     = y_frame.*win_copy;
    y_out_frame = y_frame(1:frame_size,1:channel)+y_frame_old;
    y_frame_old = y_frame(frame_size+1:end,1:channel);
    %------------------------OUT----------------------------%
    y_out_all(pos:pos+frame_size-1,:)   = y_out_frame;
end


fprintf('!!!!!!!done!!!!!!!!!!!!!'); 

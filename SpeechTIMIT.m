clc
tic
sentence_type = 'SA1' ;
main_path = 'D:\TIMITDIC data_split_matlab_SA1_DR25_M';

% sentence_type = 'SXI' ;
% main_path = 'D:\TIMITDIC data_split_matlab_SX_DR25_M';
% main_path = 'D:\TIMITDIC data_split_matlab_SX_DR25_M_butter';

threshold = 0.03 ;

frameLen = 100 ;

silentMax = 300 ;

durationMin = 300 ;

img_save = false;
audio_save = false;
img_save = true;
audio_save = true;

save_limit = 3;

ext_img_save = false;
ext_audio_save = false;

for data_set = {'TEST', 'TRAIN'}
% for data_set = {'TRAIN'}

    SF_path = fullfile('D:\TIMITDIC data', data_set) ;
    NSF_path = fullfile( ...
        main_path, ...
        append('threshold', num2str(threshold), '_frameLen', num2str(frameLen), '_silentMax', num2str(silentMax)), data_set) ;

    mkdir (char(NSF_path))

    class_name_list = {'DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8'} ;
    type_name_list = {'F', 'M'};

    if contains(main_path, '_DR25')
        class_name_list = {'DR2', 'DR5'} ;
    end

    if contains(main_path, '_M')
        type_name_list = {'M'};
    end

    for type_name = type_name_list

        typ_path = fullfile(NSF_path, type_name);

        status = exist(char(typ_path), "dir") ;

        if (status == 0)
            mkdir (char(typ_path)) ;
        end

        for class_name = class_name_list
    
            CF_path = fullfile(SF_path, class_name) ;
    
            class_path = fullfile(typ_path, class_name);
    
            status = exist(char(class_path), "dir") ;
    
            if (status == 0)
                mkdir (char(class_path)) ;
            end
    
            %%%%%
            fc = 0;
            mc = 0;
            %%%%%
            
            list_idx = 1;

%             O_id_list = {};
%             O_shape_list = {};
%             O_value_list = {};

%             E_id_list = {};
%             E_shape_list = {};
%             E_value_list = {};
    
%             Z_id_list = {};
%             Z_shape_list = {};
%             Z_value_list = {};

%             SM_id_list = {};
%             SM_shape_list = {};
%             SM_value_list = {};

            files = dir(char(CF_path));
            for i = 1:length(files)
                CF_name = files(i).name;
                if ~isfolder(CF_name)
                    files2 = dir(char(fullfile(CF_path, CF_name)));
                    for j = 1:length(files2)
                        CF_name_file = files2(j).name;
                        if contains(CF_name_file, '.WAV.wav') && contains(CF_name_file, sentence_type) && upper(char(CF_name(1))) == char(type_name)
%                         if contains(CF_name_file, '.WAV.wav') && contains(CF_name_file, 'SX') && upper(char(CF_name(1))) == char(type_name)
                            
%                           建立個人資料夾
%                           ex.D:\TIMITDIC data_split_matlab_SXI\threshold0.03_frameLen100_silentMax300\TEST\M\DR2\MABW0
                            CF_name_img_path = fullfile(class_path, CF_name);

                            status = exist(char(CF_name_img_path), "dir") ;

                            if (status == 0)
                                mkdir (char(CF_name_img_path)) ;
                            end

% %                           判斷有無分割合併wav檔，有則表示此人資料執行過
%                             status = exist(char(fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_', 'split_merge.wav'))), "file") ;
% 
%                             if (status ~= 0)
%                                 continue
%                             end
%                             
                            [data1, fs1] = audioread(char(fullfile(SF_path, class_name, CF_name, CF_name_file))) ;
                            
                            if contains(main_path, 'butter')
                                fs = 500;
                                x = data1;
    
    %                             define to use OP's code
                                charnnel = 1000;
    %                             design filter
                                [b_pass, a_pass] = butter(5, [1, 30] / (fs / 2), 'bandpass');
                                data2 = filtfilt(b_pass, a_pass, x);

                                data1 = data2;
    
                                if img_save
                                    fig = figure;
                                    t = tiledlayout(2,1, 'Parent', fig);
                                    
                                    nexttile
                                    plot(data1,'-') ;
                                    grid on;
    
                                    nexttile
                                    plot(data2,'-') ;
                                    grid on;
                                end
                            end

                            %%%%%
%                             if CF_name(1) == 'F'
%                                 if fc >= 200
%                                     continue
%                                 end
%                                 fc = fc + 1;
%                             end
%     
%                             if CF_name(1) == 'M'
%                                 if mc >= 200
%                                     continue
%                                 end
%                                 mc = mc + 1;
%                             end
                            %%%%%
    
                            SPH = [] ;
                            T = [];
                            E = [];
%                             Z = [];
%                             E1 = [];
    
                            ind=length(data1') ;
    
                            x=data1' ;
                            x_split = zeros(1, ind);
                            x_t = 1:1:ind;
%                             x_split_merge = [];

                            for i=1:ind-frameLen
                                sum=0 ;
                                sum1=0 ;
                                for j=1:frameLen - 1
                                    sum=sum+(x(1,i+j))^2 ;
                                    v1=1 ;
                                    v2=1 ;
                                    if (x(1,i+j)<0) 
                                        v1 =-1 ; 
                                    end
                                    if (x(1,i+j+1)<0) 
                                        v2 =-1 ; 
                                    end
                                    sum1=sum1+abs(v2-v1) ;
                                end
                                SPH(1,i)=x(1,i) ;  % 正規化
                                T(1,i)=i ;         % 資料序
                                E(1,i)=sum ;       %
%                                 Z(1,i)=sum1 ;      %
                            end
    
                            IND=max(E) ;
                            % E1=hilbert(E) ;
                            % DE=(E-E1) ;
                            E=(E)/IND ;
    
                            % IND=max(E) ;
                            % E1=hilbert(E) ;
                            % DE=(E-E1) ;
                            % E=(E-DE)/IND ;
    
                            IND=max(SPH) ;
                            SPH=SPH/IND ;
    
                            ind=length(E) ;
                            c=1 ;
                            for i=1:ind
                                pp(1,c)=0 ;
                                if (E(1,i)>=threshold)
                                    pp(1,c)=i ;
                                end
                                c=c+1 ;
                            end
    
                            for i=1:ind
                                E1(1,i)=0 ;
                                if (pp(1,i)>0)
                                    E1(1,i)=SPH(1,i) ;
                                end
                            end
                            
    %                         mkdir (char(fullfile(NSF_path, class_name)), CF_name) ;
                            if img_save
%                                 figure;
                                figure('Visible', 'off');
    
                                plot(T,SPH,'-') ;
                                grid on,
                                saveas(gcf, char(fullfile(class_path, append('SPH_', strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '.png'))));
                                close(gcf);
        
                                figure('Visible', 'off');
    
                                plot(T,E,'-') ;
        %                         xlim([0, 60000]); % 將X軸範圍設定為0到60000
                                grid on,
                                saveas(gcf, char(fullfile(class_path, append('E_', strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '.png'))));
                                close(gcf);
    
                                if ext_audio_save
                                    audiowrite(char(fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_E.wav'))), E, fs1) ;
%                                     audiowrite(char(fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_Z.wav'))), Z, fs1) ;
                                end

%                                 figure('Visible', 'off');
%     
%                                 plot(T,Z,'-') ;
%         %                         xlim([0, 60000]); % 將X軸範圍設定為0到60000
%         %                         ylim([0, 185]); % 將Y軸範圍設定為0到185
%                                 grid on,
%                                 saveas(gcf, char(fullfile(class_path, append('Z_', strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '.png'))));
%                                 close(gcf);
%         
                                figure('Visible', 'off');
    
                                plot(T,E1,'-') ;
                                grid on,
                                saveas(gcf, char(fullfile(class_path, append('E1_', strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '.png'))));
                                close(gcf);
                            end
                            
%                             O_id_list{list_idx} = append(CF_name, '_', strrep(CF_name_file, '.WAV.wav', ''));
%                             O_shape_list{list_idx} = length(data1);
%                             O_value_list{list_idx} = data1;

%                             能量和零交越 
%                             E_id_list{list_idx} = append(CF_name, '_', strrep(CF_name_file, '.WAV.wav', ''));
%                             E_shape_list{list_idx} = length(E);
%                             E_value_list{list_idx} = E;

%                             Z_id_list{list_idx} = append(CF_name, '_', strrep(CF_name_file, '.WAV.wav', ''));
%                             Z_shape_list{list_idx} = length(Z);
%                             Z_value_list{list_idx} = Z;
                            
%                             list_idx = list_idx + 1;
                            
                            if 1 == 1

%                                 CF_name_img_path = fullfile(NSF_path, class_name, CF_name);
%         
%                                 status = exist(char(CF_name_img_path), "dir") ;
%                         
%                                 if (status == 0)
%                                     mkdir (char(CF_name_img_path)) ;
%                                 end
                                
                                startPoint = 0 ;
                                endPoint = 0 ;
                                
                                silentCount = 0 ;
        
                                splitCount = 0 ;
        
                                txt_path = fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_', '.txt')) ;
                                
                                % 開啟文件
                                txtID = fopen(char(txt_path), 'w+');

                                pointList = [];
        
                                for i=1:length(T)
                                    if (startPoint == 0) && (E1(1,i) ~= 0)
                                        startPoint = i ;
                                    end
        
                                    if (startPoint ~= 0)
                                        if (E1(1,i) == 0)
                                            silentCount = silentCount + 1 ;
                                        else
                                            silentCount = 0 ;
                                        end
                                    end
        
                                    if silentCount >= silentMax
                                        endPoint = i - silentMax ;
                                    end
        
                                    if (startPoint ~= 0) && (endPoint ~= 0) && (endPoint - startPoint > durationMin)
                                        splitCount = splitCount + 1 ;
                                        
                                        figure('Visible', 'off');
                                        
                                        if img_save
                                            plot(T(startPoint:endPoint),E1(startPoint:endPoint),'-') ;
                                            grid on,
                                            saveas(gcf, char(fullfile(CF_name_img_path, append('E1_', strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_', num2str(splitCount), '.png'))));
                                            close(gcf);
            
                                            figure('Visible', 'off');
            
                                            plot(T(startPoint:endPoint),x(startPoint:endPoint),'-') ;
                                            grid on,
                                            saveas(gcf, char(fullfile(CF_name_img_path, append('x_', strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_', num2str(splitCount), '.png'))));
                                            close(gcf);
                                        end

%                                         x_split(startPoint:endPoint) = x(startPoint:endPoint) ;
%                                         x_split_merge = [x_split_merge, x(startPoint:endPoint)];
        
                                        % 寫入資料
                                        fprintf(txtID, '---------------------------------%d\n', splitCount);
                                        fprintf(txtID, '%f\n', endPoint - startPoint);
                                        fprintf(txtID, '%f:%f\n', startPoint, endPoint);

                                        pointList = [pointList; [startPoint, endPoint]];
        
                                        if audio_save
                                            audiowrite(char(fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_', num2str(splitCount), '.wav'))), x(startPoint:endPoint), fs1) ;
                                        end

                                        startPoint = 0 ;
                                        endPoint = 0 ;
                                        silentCount = 0 ;
                                    end
        
                                end
                                 
%                                 呈現原始訊號與切割後訊號
                                if 1 == 2
                                    fig = figure;
                                    surf(peaks);
    %                                 fig = figure('Visible', 'off');
    
%                                     t = tiledlayout(5,1, 'Parent', fig);
                                    t = tiledlayout(4,1, 'Parent', fig);
                                    
                                    nexttile
                                    plot(x_t,x,'b-') ;
                                    grid on;
                                    
                                    nexttile
                                    plot(x_t,x,'b-',x_t,x_split,'r-') ;
                                    grid on;
                                    
                                    nexttile
                                    plot(x_t,x_split,'r-') ;
                                    grid on;

                                    nexttile
                                    plot(T,E,'-') ;
                                    grid on;
                                    
%                                     nexttile
%                                     plot(x_split_merge,'-') ;
%                                     xlim([0, length(x_t)]);
%                                     grid on;
    
                                    savefig(char(fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '.fig'))));
    
%                                     set(fig, 'Visible', 'on');
%                                     saveas(fig, char(fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '.fig'))));
%                                     close(fig);
                                end
                                
%                                 if ext_audio_save
%                                     audiowrite(char(fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_', 'split_merge.wav'))), x_split_merge, fs1) ;
%                                 end
%                                     
%                                 SM_id_list{list_idx} = append(CF_name, '_', strrep(CF_name_file, '.WAV.wav', ''));
%                                 SM_shape_list{list_idx} = length(x_split_merge);
%                                 SM_value_list{list_idx} = x_split_merge;
%         
                                % 關閉文件
                                fclose(txtID);

                                maxCount = 0;

                                while maxCount < 3
%                                   取得E最大值

                                    [max_value, linear_index] = max(E(:));
                                   
%                                   看最大值落在哪個區間
                                    for i=1:length(pointList)
                                        if linear_index >= pointList(i, 1) && linear_index <= pointList(i, 2)
                                            maxCount = maxCount + 1;
                                            audiowrite(char(fullfile(CF_name_img_path, append(strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_E_max_', num2str(maxCount), '.wav'))), x(pointList(i, 1):pointList(i, 2)), fs1) ;
                                            
%                                           驗證用
                                            plot(T(pointList(i, 1):pointList(i, 2)),x(pointList(i, 1):pointList(i, 2)),'-') ;
                                            grid on,
                                            saveas(gcf, char(fullfile(CF_name_img_path, append('x_', strrep(CF_name_file, '.WAV.wav', ''), '_', CF_name, '_E_max_', num2str(maxCount), '.png'))));
                                            close(gcf);

                                            E(pointList(i, 1):pointList(i, 2)) = 0;

                                            break;
                                        end
                                    end
                                end
                            end

                            list_idx = list_idx + 1;
    
                        end
                    end
                end
            end

%             % 将结构体转换为 JSON 格式
%             jsonStructE = struct();
%             jsonStructE.id_list = E_id_list;
%             jsonStructE.shape_list = E_shape_list;
%             jsonStructE.value_list = E_value_list;
% 
%             jsonStringE = jsonencode(jsonStructE, PrettyPrint=true);
% 
%             % 将 JSON 字符串保存到文件中
%             fid = fopen(fullfile(char(class_path), 'E.json'), 'w');
%             fprintf(fid, jsonStringE);
%             fclose(fid);
% 
%             % 将结构体转换为 JSON 格式
%             jsonStructZ = struct();
%             jsonStructZ.id_list = Z_id_list;
%             jsonStructZ.shape_list = Z_shape_list;
%             jsonStructZ.value_list = Z_value_list;
% 
%             jsonStringZ = jsonencode(jsonStructZ, PrettyPrint=true);
% 
%             % 将 JSON 字符串保存到文件中
%             fid = fopen(fullfile(char(class_path), 'Z.json'), 'w');
%             fprintf(fid, jsonStringZ);
%             fclose(fid);
% 
%             % 将结构体转换为 JSON 格式
%             jsonStructSM = struct();
%             jsonStructSM.id_list = SM_id_list;
%             jsonStructSM.shape_list = SM_shape_list;
%             jsonStructSM.value_list = SM_value_list;
% 
%             jsonStringSM = jsonencode(jsonStructSM, PrettyPrint=true);
% 
%             % 将 JSON 字符串保存到文件中
%             fid = fopen(fullfile(char(class_path), 'SM.json'), 'w');
%             fprintf(fid, jsonStringSM);
%             fclose(fid);
% 
%             % 将结构体转换为 JSON 格式
%             jsonStructO = struct();
%             jsonStructO.id_list = O_id_list;
%             jsonStructO.shape_list = O_shape_list;
%             jsonStructO.value_list = O_value_list;
% 
%             jsonStringO = jsonencode(jsonStructO, PrettyPrint=true);
% 
%             % 将 JSON 字符串保存到文件中
%             fid = fopen(fullfile(char(class_path), 'O.json'), 'w');
%             fprintf(fid, jsonStringO);
%             fclose(fid);
        end
    end
end

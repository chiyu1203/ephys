%% Amalia - UCL - 14/08/2019
%% SS - UCL - 31/12/2024
%% CL Ukonstanz 04/08/2025
function calibration_file(input_dir,input_file_name,output_file_name)
%% manually load the path of file for intensity for different colours
arguments
   input_dir (1,1) string= 'C:\Users\neuroPC\Documents\GitHub\UnityDataAnalysis\';
   input_file_name (1,1) string= 'rgb_scale_4_gamma_calibration.csv';
   output_file_name (1,1) string= 'gamma_corrected_LUT.bmp';   
end
this_file_path = fullfile(input_dir,input_file_name);
m=readmatrix(this_file_path);
PixelValue=m';

%% manually input measured intensity for different colours
% values measured on LED panel 20241127
% PixelValue(:,1) = [7277.35639995 7291.27643963 7275.8289362 7277.63516328 7267.03253691 7283.52112969 7345.82732681 7304.3610085 7319.63853078 7359.03869235 7390.86371332]; % red
% PixelValue(:,2) = [7420.69100728 7798.59305875 8238.128769 9072.65530129 10369.0955544 12362.43067711 14959.65440714 18403.08369213 22196.95268774 26808.98945377 26811.76513828]; % green
% PixelValue(:,3) = [7604.25046542 8259.84202143 9067.06483285 10534.98855641 12800.16852187 16387.50763945 20965.74331547 27064.99734051 33715.82242941 40530.90949651 40516.78559366]; % blue
% PixelValue(:,4) = [7572.91992003 8816.75315728 9616.52770525 12502.64839149 14672.43924399 21684.50570624 25987.45886493 38407.6765844 44699.82081537 61737.45651771 61725.54245267]; % gray
% values measured on LED panel 20241104
% PixelValue(:,1) = [99.1554101 98.7550191 99.44226479 99.13503817 99.72983933 99.95603315 100.41627964 101.14439509 101.47652009 102.83541708 102.66381991]; % red
% PixelValue(:,2) = [83.82162732 88.20044984 96.80505356 110.99890214 133.25215891 164.64457829 209.6954061 265.66840424 337.53137391 422.26202214 515.68464232]; % green
% PixelValue(:,3) = [91.88249932 99.10043438 115.46287579 139.39662485 180.12467177 235.59176812 318.04036588 416.89953279 546.58091528 698.17913054 865.55278858]; % blue
% PixelValue(:,4) = [73.87377757 86.99856277 110.8829196 150.73163798 215.13504085 302.15964614 427.70755512 583.178667 784.78858711 1017.79326141 1281.91026667]; % gray



%% compute inverse function for each colour
scale = 0:0.1:1;
%[RGB_fit,RGB_params] = InverseGamma(PixelValue,scale','exp');
[RGB_fit,~] = InverseGamma(PixelValue,scale','pm_exp');

%% in parallel compute a 'half intensity' output
% This is because it may be more useful to simply use half the available
% luminance range, rather than the full range
% To do this we can simply linearly interpolate the first 128 values in
% each column
RGB_fit_half = NaN(size(RGB_fit));
RGB_fit_xval = 0:255;
RGB_fit_half_xval = 0:0.5:127.5;
for thisColor = 1:size(RGB_fit,2)
    RGB_fit_half(:,thisColor) = interp1(RGB_fit_xval,RGB_fit(:,thisColor),RGB_fit_half_xval,'spline');
end
% Show the two
figure('Name','Half-intensity and full intensity outputs')
titles = {'R','G','B','K'};
for thisColor = 1:size(RGB_fit,2)
    subplot(2,2,thisColor)
    plot(RGB_fit_xval,RGB_fit(:,thisColor),'b-'); hold on
    plot(RGB_fit_xval,RGB_fit_half(:,thisColor),'c-'); hold on
    title([titles{thisColor}, 'b = original, c = new'])
end


%% Write exponential fits to .bmp 
imwrite(RGB_fit(:,1:3)',fullfile(input_dir,output_file_name));
imwrite(RGB_fit_half(:,1:3)',fullfile(input_dir,strcat('half_',output_file_name)));

%% Save RGB part of inverse function to use with psychtoolbox
% Output table to use with Screen(LoadNormalizedGammaTable) function
table=RGB_fit(:,1:3);

%% Post-correction measurement

%% compare results
figure
subplot(221)
plot(scale,PixelValue(:,1),'r:')
hold on
% plot(scale,PixelValue_post(:,1),'r')
subplot(222)
plot(scale,PixelValue(:,2),'g:')
hold on
% plot(scale,PixelValue_post(:,2),'g')
subplot(223)
plot(scale,PixelValue(:,3),'b:')
hold on
% plot(scale,PixelValue_post(:,3),'b')
end
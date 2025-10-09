% GammaCorrectionSnippet.m
% Example LUT calibration (needs 'optimization' toolbox)
% History
%       Unknown upload from GammaCalibrationSnippet
%       SGS 13/6/2019 - updated to correct
%       SGS 1/09/2024 - updated make more legible and to add pm_exp output

function [dummyRGB, dummyParams] = InverseGamma(PixelValueIN,xaxisForPixelValue,outputFlag)
if nargin < 3
    outputFlag = 'exp'; % hyp, exp, or pm_exp
end
if nargin < 2
    xaxisForPixelValue = linspace(0,1,size(PixelValueIN,1))'; % NB assume 0-1 in equal steps
    warning('Assuming calibration table was obtained for values in range 0-1 in equal based steps')
end

% For each colour in the gamma table
for i = 1:size(PixelValueIN,2)

    % Get data
    PixelValue = PixelValueIN(:,i);

    switch(lower(outputFlag))
        case 'exp'
            % Power - The inverse exponential is very easy...you just make n into 1/n
            exp_fun = @(c,xdata) c(1).*(xdata.^c(2))+c(3);

            % Fit with exponential
            init =  [10 2 1];
            SSE = @(c) sum((PixelValue - exp_fun(c,xaxisForPixelValue)).^2);
            f_exp = fminsearch(SSE,init);

            % Calculate normalized function [NB we set gain to 1, raise to the minus power and remove the offset]
            interp_xaxisForPixelValue = linspace(min(xaxisForPixelValue),max(xaxisForPixelValue),256); % Use size of desired LUT to set interpolation vals
            return_xaxisForPixelValue = interp_xaxisForPixelValue;
            interp_data = exp_fun(f_exp,interp_xaxisForPixelValue);
            norm_data =  exp_fun([1 f_exp(2) 0],interp_xaxisForPixelValue);
            inverse_data =  exp_fun([1 1/f_exp(2) 0],return_xaxisForPixelValue);
            checkVals = exp_fun([1 f_exp(2) 0],inverse_data);
            reportText = sprintf('\nExponential: \nGain %3.2f\nExp %3.4f\nOffset %3.4f',f_exp);

            % Save for later
            dummyRGB(:,i) = inverse_data;
            dummyParams(:,i) = f_exp;

        case 'hyp'
            % The inverse of a sigmoid is trickier [I derived this and I am not completely sure this is correct but I hope you get the idea]
            hyp_fun = @(c,xdata) c(1).*((xdata.^c(2))./((xdata.^c(2))+(c(3).^c(2))))+c(4);
            inverse_hyp_fun =  @(c,xdata)  (-1*(xdata.*(c(3).^c(2)))./(xdata-1)).^(1/c(2));

            % Fit with sigmoid
            init =  [10 2 0.5 1];
            SSE = @(c) sum((PixelValue - hyp_fun(c,xaxisForPixelValue)).^2);
            f_hyp = fminsearch(SSE,init);

            % Calculate normalized function
            interp_xaxisForPixelValue = linspace(min(xaxisForPixelValue),max(xaxisForPixelValue),256); % Use size of LUT to set interpolation vals
            return_xaxisForPixelValue = interp_xaxisForPixelValue; % generate LUT LUT to get interpolation vals
            interp_data = hyp_fun(f_hyp,interp_xaxisForPixelValue);
            norm_data =  hyp_fun([1 f_hyp(2) f_hyp(3) 0],interp_xaxisForPixelValue);
            inverse_data = inverse_hyp_fun([1 f_hyp(2) f_hyp(3) 0],return_xaxisForPixelValue);
            checkVals = hyp_fun([1 f_hyp(2) f_hyp(3) 0],inverse_data);
            reportText = sprintf('\nSigmoid: \nGain %3.2f\nExp %3.4f\nInflection %3.4f\nOffset %3.4f',f_hyp);

            % Save for later
            dummyRGB(:,i) = inverse_data;
            dummyParams(:,i) = f_hyp;

        case 'pm_exp'
            % exp
            exp_fun = @(c,xdata) c(1).*(c(2).^xdata)+c(3);

            % Fit with exponential then use manual interpolation to extract
            % values
            init =  [10 2 1];
            SSE = @(c) sum((PixelValue - exp_fun(c,xaxisForPixelValue)).^2);
            f_exp = fminsearch(SSE,init);

            % Get highly interpolated data
            interp_xaxisForPixelValue = linspace(min(xaxisForPixelValue),max(xaxisForPixelValue),256*8); % subsample LUT to get interpolation vals
            return_xaxisForPixelValue = linspace(min(xaxisForPixelValue),max(xaxisForPixelValue),256);
            interp_data = exp_fun(f_exp,interp_xaxisForPixelValue);

            % We want to find the x vals that correspond to linear output along range
            return_yaxisForPixelValue = linspace(min(interp_data),max(interp_data),256);
            inverse_data = [];
            for kk = 1:length(return_yaxisForPixelValue) % For each return value, find nearest neighbour
                [~,tt] = min(abs(interp_data-return_yaxisForPixelValue(kk)));
                inverse_data(kk) = interp_xaxisForPixelValue(tt);
            end
            norm_data =  exp_fun([1 f_exp(2) 0],return_xaxisForPixelValue);
            checkVals = exp_fun([1 f_exp(2) 0],inverse_data);
            reportText = sprintf('\nExponential: \nGain %3.2f\nExp %3.4f\nOffset %3.4f',f_exp);

            % Save for later
            dummyRGB(:,i) = inverse_data;
            dummyParams(:,i) = f_exp;

    end

    % Plot results
    LineColor = [0 0 0];
    LineColor(i) = 1;

    figure
    subplot(221)
    h1 = plot(xaxisForPixelValue,PixelValue,'o','Color',LineColor); hold on
    h2 = plot(interp_xaxisForPixelValue,interp_data,'-','Color',LineColor);
    text(0.05,0.95,reportText,'Units','normalized')
    title('Original')

    subplot(222)
    plot(return_xaxisForPixelValue,norm_data,'-','Color',LineColor); hold on
    title('Normalized')

    subplot(223)
    plot(return_xaxisForPixelValue,inverse_data,'-','Color',LineColor); hold on
    title('Inverse')

    subplot(224)
    plot(return_xaxisForPixelValue,checkVals,'-','Color',LineColor); hold on
    title('Check of correction')

end

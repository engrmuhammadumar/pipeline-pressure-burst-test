% Initialize variables for coherogram feature pool and labels
featurePool = [];
labels = [];

% Define the desired coherogram image size for CNN input
imageSize = [224, 224];

datasets = {'Impeller (3.0BAR)', 'Mechanical seal Hole (3BAR)', 'Mechanical seal Scratch (3.0BAR)', 'Normal (3BAR)'};
fs = 46000;
step = 1/fs;
t = linspace(0, 1, fs);
load('h_channel1_baseline.mat');
time = 1;

for i = 3:length(datasets)
    Folder = datasets{i};
    
    S = dir(fullfile(Folder, '*.mat'));
    for k = 1:numel(S) 
        F = fullfile(Folder, S(k).name); % you need FOLDER here too.
        data = load(F);
        mat = data.signal;
        
        s = mat(3, :);
        s = lowpass(s, 46000, fs);
        [Wcoh, ~, ~, F] = wcoherence(h_channel1_baseline, s); % Perform wavelet coherence analysis
        
        % Plot the coherogram
        figure;
        imagesc(t, F, abs(Wcoh));
        axis xy;
        xlabel('Time (seconds)');
        ylabel('Frequency (Hz)');
        colormap('jet');
        colorbar;
        title('Wavelet Coherence');

        % Resize the coherogram image for CNN input
        coherogramImage = imresize(abs(Wcoh), imageSize);

        % Store the coherogram image in the feature pool
        featurePool = cat(4, featurePool, coherogramImage);

        % Store the label for the current coherogram image based on the dataset
        if strcmp(datasets{i}, 'Impeller (3.0BAR)')
            labels = [labels; 1];
        elseif strcmp(datasets{i}, 'Mechanical seal Hole (3BAR)')
            labels = [labels; 2];
        elseif strcmp(datasets{i}, 'Mechanical seal Scratch (3.0BAR)')
            labels = [labels; 3];
        elseif strcmp(datasets{i}, 'Normal (3BAR)')
            labels = [labels; 4];
        end
        
        % Save the coherogram image
        saveas(gcf, fullfile('D:\Niamat khattak\CP Project\Vibration\Images Dataset\', datasets{i}, strcat('Image', num2str(k), '.png')));
        
        close; % Close the figure to prevent unnecessary memory usage
    end
end

% Save the coherogram feature pool and labels for CNN training
save('coherogram_feature_pool.mat', 'featurePool', '-v7.3');
save('coherogram_labels.mat', 'labels'); 

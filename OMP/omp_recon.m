% Main Script
% Load image
originalImage = imread('autumn.tif');

% Convert image to grayscale if it's RGB
if size(originalImage, 3) == 3
    originalImage = rgb2gray(originalImage);
end
originalImage = imresize(originalImage, [100 100]);

% Display original image
figure;
subplot(1, 3, 1);
imshow(originalImage, []);
title('Original Image');

% Parameters
compressionRatio = 0.5;
[M, N] = size(originalImage);
k = round(compressionRatio * M * N);

% Generate a random sensing matrix (measurement matrix)
A = randn(k, M * N);

% Normalize columns of A
A = A ./ sqrt(sum(A.^2));

% Perform DCT on the entire image
dctImage = dct2(double(originalImage));

% Vectorize DCT coefficients
x = dctImage(:);

% Normalize x
x = x ./ norm(x);

% Compressed Sensing: Acquire measurements
y = A * x;

% OMP Reconstruction
sparsity = 500;
reconstructedSignal = omp_reconstruction(y, A, sparsity);

% Reshape the reconstructed signal to an image
reconstructedImage = reshape(reconstructedSignal, M, N);

% Apply IDCT to obtain the reconstructed image
reconstructedImage = idct2(reconstructedImage);

% Normalize reconstructed image for display
reconstructedImage = (reconstructedImage - min(reconstructedImage(:))) / (max(reconstructedImage(:)) - min(reconstructedImage(:)));

% Display the original, measurements, and reconstructed images
subplot(1, 3, 2);
%imshow(reshape(A * x, M, N), []);
title('Compressed Measurements');

subplot(1, 3, 3);
imshow(reconstructedImage, []);
title('Reconstructed Image');


originalImage = im2double(originalImage);
reconstructedImage = im2double(reconstructedImage);

% Calculate PSNR in dB
psnrValue = psnr(originalImage, reconstructedImage);

% Display the PSNR value
disp(['PSNR: ', num2str(psnrValue), ' dB']);

% OMP Reconstruction Function
function reconstructedSignal = omp_reconstruction(measurements, sensingMatrix, sparsity)
    [M, N] = size(sensingMatrix);
    reconstructedSignal = zeros(N, 1);
    residual = measurements;

    for k = 1:sparsity
        innerProducts = abs(sensingMatrix' * residual);
        [~, index] = max(innerProducts);
        supportSet(k) = index;
        x_hat = pinv(sensingMatrix(:, supportSet(1:k))) * measurements;
        residual = measurements - sensingMatrix(:, supportSet(1:k)) * x_hat;
    end

    reconstructedSignal(supportSet) = x_hat;
end

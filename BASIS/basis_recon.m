
% Load image
originalImage = imread('cameraman.tif');


% Convert image to grayscale if it's RGB
if size(originalImage, 3) == 3
    originalImage = rgb2gray(originalImage);
end
originalImage = imresize(originalImage,[100 100]);
% Parameters
imshow(originalImage);
title('Original Image');

 compressionRatio = 0.3; % Adjust compression ratio as needed
 [M, N] = size(originalImage);
 k = round(compressionRatio * M * N); % Number of measurements
 
 % Generate a random sensing matrix (measurement matrix)
 A = randn(k, M*N);
 
 % Perform DCT on the entire image
 dctImage = dct2(double(originalImage));
 
 % Vectorize DCT coefficients
 x = dctImage(:);
 
 % Compressed Sensing: Acquire measurements
y = A * x;



% Reconstruction using Basis Pursuit with cvx
cvx_begin
    variable reconstructedX(M*N, 1);
    minimize(norm(reconstructedX, 1));
    subject to
        A * reconstructedX == y;
cvx_end

% Reshape the reconstructed vector to a matrix
reconstructedImage = idct2(reshape(reconstructedX, M, N));

% Display the reconstructed image
figure;
imshow(uint8(reconstructedImage));
title('Reconstructed Image');

% Calculate Peak Signal-to-Noise Ratio (PSNR)
psnrValue = psnr(uint8(originalImage), uint8(reconstructedImage));
fprintf('PSNR: %.2f dB\n', psnrValue);

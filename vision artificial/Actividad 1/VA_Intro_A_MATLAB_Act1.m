%% VISIÓN ARTIFICIAL · PRÁCTICA PASO A PASO EN MATLAB
% Equivalente al notebook VA_Alumnado_Colab_Paso_a_Paso_CORREGIDO.ipynb.
% Ejecuta las secciones (%%) en orden usando «Run Section».
% Necesita Image Processing Toolbox para rgb2gray e imshow.
% La imagen se selecciona desde el PC; no se genera ninguna alternativa.

%% 1. Preparar el entorno
% MATLAB ya incluye las operaciones matriciales necesarias.
% Comprobamos la versión y limpiamos las figuras y variables anteriores.
clearvars; close all; clc;
disp(['Versión de MATLAB: ', version]);
% En la Command Window puedes consultar la ayuda: help imread

%% 2. Subir / abrir una imagen
% Se abre el explorador de archivos para elegir una imagen PNG o JPG.
[nombre, carpeta] = uigetfile({'*.png;*.jpg;*.jpeg', ...
    'Imágenes PNG/JPEG (*.png, *.jpg, *.jpeg)'}, 'Selecciona una imagen');
if isequal(nombre, 0)
    error('No se seleccionó ninguna imagen. Ejecuta esta sección de nuevo.');
end
imagen = imread(fullfile(carpeta, nombre));

% Convertimos la imagen a RGB si se cargó en escala de grises.
% Si se cargó una imagen RGBA, descartamos el canal alfa.
if ndims(imagen) == 2
    I_rgb = repmat(imagen, [1, 1, 3]);
elseif size(imagen, 3) >= 3
    I_rgb = imagen(:,:,1:3);
else
    error('Formato de imagen no compatible.');
end
fprintf('Archivo: %s | Dimensiones: %s | Tipo: %s\n', ...
    nombre, mat2str(size(I_rgb)), class(I_rgb));
figure; imshow(I_rgb); title('Imagen original');

%% 3. Inspeccionar y convertir
% Convertimos la imagen RGB cargada desde el PC a escala de grises.
% I_gray tiene valores de intensidad de 0 a 255 para imágenes uint8.
% im2double convierte imágenes uint8 al intervalo [0, 1].
I_gray = rgb2gray(I_rgb);
I = im2double(I_gray);
fprintf('RGB: %s, %s\n', mat2str(size(I_rgb)), class(I_rgb));
fprintf('Grises: %s, %s\n', mat2str(size(I_gray)), class(I_gray));
fprintf('Normalizada: %s, %s, rango: %.4f a %.4f\n', ...
    mat2str(size(I)), class(I), min(I(:)), max(I(:)));
figure; imshow(I_gray); title('Imagen en escala de grises');

%% 4. Brillo y saturación
% Sumamos 80 niveles de intensidad, con recorte explícito en 255.
% Convertimos antes a double para no depender de la saturación de uint8.
delta = 80;
J = uint8(min(max(double(I_gray) + delta, 0), 255));
saturacion = 100 * nnz(J == 255) / numel(J);
fprintf('Píxeles saturados: %.2f%%\n', saturacion);
figure;
subplot(1,2,1); imshow(I_gray); title('Original');
subplot(1,2,2); imshow(J); title(sprintf('Brillo +%d', delta));
% Prueba: cambia delta de 80 a 30 y vuelve a ejecutar esta sección.

%% 5. Histograma
% Comparamos la distribución de intensidades antes y después del brillo.
figure;
subplot(1,2,1);
histogram(double(I_gray(:)), 0:256);
title('Original'); xlabel('Intensidad'); ylabel('N.º de píxeles');
xlim([0 256]);
subplot(1,2,2);
histogram(double(J(:)), 0:256);
title(sprintf('Brillo +%d', delta));
xlabel('Intensidad'); ylabel('N.º de píxeles');
xlim([0 256]);

%% 6. Negativo y gamma
% Usamos la imagen I normalizada al intervalo [0,1].
negativo = 1 - I;
gamma_clara = I .^ 0.4;
gamma_oscura = I .^ 2;
figure;
subplot(1,3,1); imshow(negativo); title('Negativo');
subplot(1,3,2); imshow(gamma_clara); title('Gamma 0.4');
subplot(1,3,3); imshow(gamma_oscura); title('Gamma 2');
% Pregunta: ¿qué gamma aclara las zonas oscuras?

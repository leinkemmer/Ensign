% rk = 5
% x -> 64
% v -> 256

clc
clear all
close all

fileID1 = fopen('el_energy_order1_1d.txt');
fileID2 = fopen('el_energy_gpu_order1_1d.txt');

fileID3 = fopen('err_mass_order1_1d.txt');
fileID4 = fopen('err_mass_gpu_order1_1d.txt');

fileID5 = fopen('err_energy_order1_1d.txt');
fileID6 = fopen('err_energy_gpu_order1_1d.txt');

A = fscanf(fileID1,'%f');

tstar = A(1);
tau = A(2);
ttt = tau:tau:tstar;
el_energy = A(3:end);

B = fscanf(fileID2,'%f');
el_energy_gpu = B(3:end);

figure
subplot(2,3,1)
semilogy(ttt,el_energy,'-r','Linewidth',2)
hold on
semilogy(ttt,el_energy_gpu,'--b','Linewidth',2)

semilogy(ttt,1/1000*exp(-0.306*ttt))
title('Electric energy - Order 1')
legend('CPU','GPU')

err_mass = fscanf(fileID3,'%f');
err_mass_gpu = fscanf(fileID4,'%f');

subplot(2,3,2)
semilogy(ttt,err_mass,'-r','Linewidth',2)
hold on
semilogy(ttt,err_mass_gpu,'--b','Linewidth',2)
title('Err mass - Order 1')
legend('CPU','GPU')

err_energy = fscanf(fileID5,'%f');
err_energy_gpu = fscanf(fileID6,'%f');
subplot(2,3,3)
semilogy(ttt,err_energy,'-r','Linewidth',2)
hold on
semilogy(ttt,err_energy_gpu,'--b','Linewidth',2)
title('Err energy - Order 1')
legend('CPU','GPU')


fileID1 = fopen('el_energy_order2_1d.txt');
fileID2 = fopen('el_energy_gpu_order2_1d.txt');

fileID3 = fopen('err_mass_order2_1d.txt');
fileID4 = fopen('err_mass_gpu_order2_1d.txt');

fileID5 = fopen('err_energy_order2_1d.txt');
fileID6 = fopen('err_energy_gpu_order2_1d.txt');

A = fscanf(fileID1,'%f');

tstar = A(1);
tau = A(2);
ttt = tau:tau:tstar;
el_energy = A(3:end);

B = fscanf(fileID2,'%f');
el_energy_gpu = B(3:end);

subplot(2,3,4)
semilogy(ttt,el_energy,'-r','Linewidth',2)
hold on
semilogy(ttt,el_energy_gpu,'--b','Linewidth',2)

semilogy(ttt,1/1000*exp(-0.306*ttt))
title('Electric energy - Order 2')
legend('CPU','GPU')

err_mass = fscanf(fileID3,'%f');
err_mass_gpu = fscanf(fileID4,'%f');

subplot(2,3,5)
semilogy(ttt,err_mass,'-r','Linewidth',2)
hold on
semilogy(ttt,err_mass_gpu,'--b','Linewidth',2)
title('Err mass - Order 2')
legend('CPU','GPU')

err_energy = fscanf(fileID5,'%f');
err_energy_gpu = fscanf(fileID6,'%f');
subplot(2,3,6)
semilogy(ttt,err_energy,'-r','Linewidth',2)
hold on
semilogy(ttt,err_energy_gpu,'--b','Linewidth',2)
title('Err energy - Order 2')
legend('CPU','GPU')

clc
clear all
close all

fileID1 = fopen('error_order1_2d.txt');
fileID2 = fopen('error_order2_2d.txt');

A = fscanf(fileID1,'%f');

ll = A(1);
for i =1:ll
  nspan(i) = A(i+1);
end
err = A(ll+2:end);

B = fscanf(fileID2,'%f');

err2 = B(ll+2:end);

figure
loglog(nspan,err(end)*(nspan/nspan(end)).^(-1),'-r','linewidth',1.5);
hold on
loglog(nspan,err,'x','linewidth',2)
hold on
loglog(nspan,err2(end)*(nspan/nspan(end)).^(-2),'-g','linewidth',1.5);
hold on
loglog(nspan,err2,'o','linewidth',2)
legend('Error order 1','Order 1','Error order 2','Order 2')

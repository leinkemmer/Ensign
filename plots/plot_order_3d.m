clc
clear all
close all

fileID1 = fopen('error_order1_3d.txt');
fileID2 = fopen('error_order2_3d.txt');

A = fscanf(fileID1,'%f');

ll = A(1);
for i =1:ll
  nspan(i) = A(i+1);
end
err = A(ll+2:end);

B = fscanf(fileID2,'%f');

err2 = B(ll+2:end);

figure
loglog(nspan,err(end)*(nspan/nspan(end)).^(-1),'-r','linewidth',2);
hold on
loglog(nspan,err,'x','linewidth',2)
hold on
loglog(nspan,err2(1)*(nspan/nspan(1)).^(-2),'-g','linewidth',2);
%hold on
%loglog(nspan,err2(1)*(nspan/nspan(1)).^(-1),'-b','linewidth',2);
hold on
loglog(nspan,err2,'o','linewidth',2)
legend('Error order 1','Order 1','Error order 2','Order 2')

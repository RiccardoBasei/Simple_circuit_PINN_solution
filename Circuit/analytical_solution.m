function [p1,p2,p3] = analytical_solution(time)

%% data

% resistances
R1 = 1;
R2 = 1;
R3 = 10;

% capacitances
C1 = 1e-3;
C2 = 1e-3;
C3 = 1e-3;

% current source
omega = 2*pi*2000;
amp = 10;

%% impedance evaluation in the Laplace domain
s = tf('s');
Z3_bis = R3/(1+s*C3*R3); % parallel of R3 and C3
Z2_bis = (R2+R3+s*C3*R3*R2)/(1+s*C3*R3+s*C2*R2+s*C2*R3+s^2*C2*C3*R2*R3); % parallel of C2 and the series of (R2 and Z2)
NOM = (R1 + R2 + R3) + s*(R1*R3*C3 + R1*C2*R2 + R1*C2*R3 + R2*R3*C3) + s^2*(R1*R2*R3*C2*C3);
DEN = 1 + s*(C3*R3 + C2*R2 + C1*R1 + C2*R3 + C1*R2 + C1*R3) +...
    s^2*(C1*C3*R1*R3 + C1*C2*R1*R2 + C1*C2*R1*R3 + C1*C3*R2*R3 + R2*R3*C2*C3) +...
    s^3*R1*R2*R3*C1*C2*C3;
Z1_bis = NOM/DEN; % parallel of C1 and the series of R1 and Z2

%% potential evaluation
% current expression in the Laplace domain
I = amp*(omega/(s^2+omega^2));

% first node potential
P1 = I*Z1_bis;

% second node potential
I2 = I - P1*s*C1;
P2 = I2*Z2_bis;

% third node potential
I3 = I2 - P2*s*C2;
P3 = I3*Z3_bis;

%% results in the time domain
[p1,~] = impulse(P1,time);
[p2,~] = impulse(P2,time);
[p3,~] = impulse(P3,time);
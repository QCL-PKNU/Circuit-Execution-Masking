OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cz q[1],q[0];
x q[0];
x q[1];
h q[1];
rx(1.5677223331356585) q[0];
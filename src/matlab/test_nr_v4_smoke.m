function test_nr_v4_smoke
assert(exist('nr_v4_evaluate','file')==2,'v4 truth-matching evaluator missing');
u=struct('pci',1,'payload',zeros(1,24),'power_db',0);
u(2)=struct('pci',2,'payload',ones(1,24),'power_db',-6);
truth=struct('users',u);
run=struct('outputs',struct('pci',1,'payload',zeros(1,24)));
run.outputs(2)=struct('pci',2,'payload',zeros(1,24));
m=nr_v4_evaluate(run,truth);
assert(m.recovered_users==1 && m.transmitted_users==2 && m.incorrect_outputs==1);
assert(m.weak_recovered==0 && m.weak_transmitted==1);
run.outputs(2).payload=ones(1,24);m=nr_v4_evaluate(run,truth);
assert(m.recovered_users==2 && m.weak_recovered==1 && m.all_recovered);
run.outputs(2).pci=1;rejected=false;
try,nr_v4_evaluate(run,truth);catch,rejected=true;end
assert(rejected,'Duplicate PCI outputs silently counted');
run.outputs=struct([]);truth.users=struct([]);m=nr_v4_evaluate(run,truth);
assert(m.recovered_users==0 && m.transmitted_users==0 && ~m.all_recovered);
assert(exist('run_nr_v4_smoke','file')==2,'smoke runner missing');
fprintf('NR V4 SMOKE HELPER TESTS PASS\n');
end

# python generation.py --model codellama_7b --save_path ./codellama_outputs/calculate_boundary_hallucination.json --data_path /mnt/ebs/data/CodeHalu/benchmarks/calculate_boundary_hallucination.json

python generation.py --model codellama_7b --save_path ./codellama_outputs/identification_hallucination.json --data_path /mnt/ebs/data/CodeHalu/benchmarks/identification_hallucination.json

python generation.py --model codellama_7b --save_path ./codellama_outputs/physical_constraint_hallucination.json --data_path /mnt/ebs/data/CodeHalu/benchmarks/physical_constraint_hallucination.json

python generation.py --model codellama_7b --save_path ./codellama_outputs/data_compliance_hallucination.json --data_path /mnt/ebs/data/CodeHalu/benchmarks/data_compliance_hallucination.json

python generation.py --model codellama_7b --save_path ./codellama_outputs/logic_breakdown.json --data_path /mnt/ebs/data/CodeHalu/benchmarks/logic_breakdown.json

python generation.py --model codellama_7b --save_path ./codellama_outputs/structural_access_hallucination.json --data_path /mnt/ebs/data/CodeHalu/benchmarks/structural_access_hallucination.json

python generation.py --model codellama_7b --save_path ./codellama_outputs/external_source_hallucination.json --data_path /mnt/ebs/data/CodeHalu/benchmarks/external_source_hallucination.json

python generation.py --model codellama_7b --save_path ./codellama_outputs/logic_deviation.json --data_path /mnt/ebs/data/CodeHalu/benchmarks/logic_deviation.json 

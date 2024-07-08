export TOKENIZERS_PARALLELISM=false

echo "***"
python print_eval_results.py --halu_type Calculate_Boundary_Hallucination --generation_file ./codellama_outputs/calculate_boundary_hallucination.json

echo "***"
python print_eval_results.py --halu_type Data_Compliance_Hallucination --generation_file ./codellama_outputs/data_compliance_hallucination.json

echo "***"
python print_eval_results.py --halu_type Structural_Access_Hallucination --generation_file ./codellama_outputs/structural_access_hallucination.json

echo "***"
python print_eval_results.py --halu_type Identification_Hallucination --generation_file ./codellama_outputs/identification_hallucination.json

echo "***"
python print_eval_results.py --halu_type External_Source_Hallucination --generation_file ./codellama_outputs/external_source_hallucination.json

echo "***"
python print_eval_results.py --halu_type Physical_Constraint_Hallucination --generation_file ./codellama_outputs/physical_constraint_hallucination.json

echo "***"
python print_eval_results.py --halu_type Logic_Deviation --generation_file ./codellama_outputs/logic_deviation.json

echo "***"
python print_eval_results.py --halu_type Logic_Breakdown --generation_file ./codellama_outputs/logic_breakdown.json

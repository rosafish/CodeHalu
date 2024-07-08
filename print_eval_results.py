import argparse
import json
from codehalu_gen_eval import load_generation, programming_halus

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate generations against problems.')
    
    parser.add_argument('--halu_type', 
                        type=str, 
                        required=True, 
                        help='The type of hallucination you want to evaluate.')

    parser.add_argument('--generation_file', 
                        type=str, 
                        required=True, 
                        help='File containing generations to be evaluated.')

    return parser.parse_args()

def main(args):
    _,ori_datas,_ = load_generation(args.generation_file)

    with open(f'evaluated_results/{args.halu_type}_errors_dict.json', 'r') as f:
        errors_dict = json.load(f)

    count = 0 
    for _, error_value in programming_halus[args.halu_type].items():
        try:
            count += errors_dict[error_value]['count']
        except Exception as e:
            count = count

    # print("halu_count: ",count)  
    total_count = len(ori_datas)
    print("total_count: ",total_count)
    halu_percentage = (count / total_count) * 100   
    halu_percentage = round(halu_percentage, 2)
    print(args.halu_type)   
    print("hallucination rate: ",halu_percentage) 

    correct = errors_dict['Correct']['count']
    print("correct: ",correct)
    correct_percentage = (correct / total_count) * 100
    correct_percentage = round(correct_percentage, 2)
    print("correct rate: ",correct_percentage)

if __name__ == "__main__":
    args = parse_args()
    main(args)

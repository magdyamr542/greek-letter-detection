train_length=$(cat data/training/coco.json| jq '.images | length')

test_length=$(cat data/testing/coco.json| jq '.images | length')

echo "Training images length $train_length"
echo "Testing images length $test_length"

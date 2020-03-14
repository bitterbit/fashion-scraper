from loader import ImageLoader
from model import create_model

def main():
    # Loading DATA
    train_path='traindata/train'
    test_path='traindata/test'
    train_data = ImageLoader(PATH=train_path, IMAGE_SIZE=128)
    test_data = ImageLoader(PATH=test_path, IMAGE_SIZE=128)
    train_data_x, train_data_y = train_data.load_dataset()
    test_data_x, test_data_y = test_data.load_dataset()

    print("LOAD DATA!")

    # Build Model
    model = create_model()

    # Train
    print("TRAINNIG...")
    history = model.fit(train_data_x, train_data_y, epochs=10) 

    # Test
    print("TEST...")
    history = model.evaluate(test_data_x, test_data_y) 

if __name__ == '__main__':
    main()

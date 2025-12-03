# NeuroFlow 🧠

A modern web application for machine learning model training and prediction, built with Next.js and Python. NeuroFlow provides an intuitive interface for creating, training, and deploying machine learning models with support for various algorithms including Linear Regression, Logistic Regression, and K-Nearest Neighbors (KNN).

## ✨ Features

### 🤖 Machine Learning Algorithms
- **Linear Regression**: Train models for continuous variable prediction
- **Logistic Regression**: Binary classification and probability estimation
- **K-Nearest Neighbors (KNN)**: Classification based on nearest data points

### 📊 Project Management
- Create and manage multiple ML projects
- Organize projects by algorithm type
- Edit and delete existing projects
- Track project creation timestamps
- Interactive project dashboard

### 🔐 User Authentication
- Secure user registration and login
- JWT-based authentication
- Password reset functionality
- Role-based access control
- Session management

### 💾 Data Management
- CSV file upload and processing
- Data preprocessing and cleaning
- Feature selection and engineering
- Data visualization and graphs
- Model training with custom parameters
- Real-time prediction capabilities

### 🎨 Modern UI/UX
- Dark/Light mode support
- Responsive design for all devices
- Material-Tailwind and Radix UI components
- Interactive dialogs and notifications
- Toast notifications for user feedback

## 🛠️ Tech Stack

### Frontend
- **Framework**: [Next.js 14](https://nextjs.org/) (React 18)
- **Language**: TypeScript
- **Styling**: 
  - [Tailwind CSS](https://tailwindcss.com/)
  - [Material-Tailwind](https://www.material-tailwind.com/)
  - [Radix UI](https://www.radix-ui.com/)
- **Icons**: 
  - Heroicons
  - Lucide React
  - React Icons
- **UI Components**:
  - Custom Radix UI components (Dialog, Select, Avatar, etc.)
  - Shadcn UI patterns
  - Command palette (cmdk)

### Backend
- **Framework**: Next.js API Routes
- **Database**: [MongoDB](https://www.mongodb.com/) with [Mongoose](https://mongoosejs.com/)
- **Authentication**: JWT (jsonwebtoken)
- **Password Hashing**: bcryptjs
- **Email**: Nodemailer

### Machine Learning
- **Language**: Python 3
- **Libraries**:
  - NumPy
  - Pandas
  - Scikit-learn
  - Joblib (model persistence)

### Development Tools
- **Linting**: ESLint
- **Code Formatting**: Next.js built-in formatter
- **Build Tool**: Next.js build system

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- [Node.js](https://nodejs.org/) (v18 or higher)
- [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/)
- [Python](https://www.python.org/) (v3.8 or higher)
- [MongoDB](https://www.mongodb.com/) (local or cloud instance)
- [pip](https://pip.pypa.io/) (Python package manager)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/visionjain/NeuroFlow.git
cd NeuroFlow
```

### 2. Install Node.js Dependencies
```bash
npm install
# or
yarn install
```

### 3. Install Python Dependencies
```bash
pip install numpy pandas scikit-learn joblib
```

### 4. Environment Variables
Create a `.env.local` file in the root directory with the following variables:
```env
# MongoDB Connection
MONGO_URI=your_mongodb_connection_string

# JWT Secret
TOKEN_SECRET=your_jwt_secret_key

# Email Configuration (for password reset)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_email_password

# Domain
DOMAIN=http://localhost:3000
```

### 5. Database Setup
Ensure your MongoDB instance is running and accessible with the connection string provided in `MONGO_URI`.

## 🎯 Usage

### Development Mode
Start the development server:
```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build
Build the application for production:
```bash
npm run build
npm start
# or
yarn build
yarn start
```

### Linting
Run ESLint to check code quality:
```bash
npm run lint
# or
yarn lint
```

## 📁 Project Structure

```
NeuroFlow/
├── public/               # Static files
├── Scripts/              # Python ML scripts
│   ├── linearreg.py      # Linear regression training
│   ├── predict.py        # Model prediction
│   └── read_preprocessing.py  # Data preprocessing
├── src/
│   ├── app/              # Next.js app directory
│   │   ├── api/          # API routes
│   │   │   └── users/    # User-related endpoints
│   │   ├── booking/      # Booking pages
│   │   ├── forgotpass/   # Password reset
│   │   ├── login/        # Login page
│   │   ├── profile/      # User profile
│   │   ├── project/      # Project pages
│   │   │   └── [id]/     # Dynamic project page
│   │   ├── signup/       # Registration page
│   │   ├── layout.tsx    # Root layout
│   │   ├── page.tsx      # Home page
│   │   └── globals.css   # Global styles
│   ├── components/       # React components
│   │   ├── AlgoComps/    # Algorithm components
│   │   ├── copybar/      # Copyright footer
│   │   ├── darkmode/     # Dark mode toggle
│   │   ├── home/         # Home page components
│   │   ├── loader/       # Loading spinner
│   │   ├── logoutbutton/ # Logout functionality
│   │   ├── navbar/       # Navigation bar
│   │   └── ui/           # Reusable UI components
│   ├── dbConfig/         # Database configuration
│   ├── helpers/          # Utility functions
│   ├── lib/              # Library code
│   └── models/           # Mongoose models
├── .eslintrc.json        # ESLint configuration
├── .gitignore            # Git ignore rules
├── components.json       # Component configuration
├── next.config.mjs       # Next.js configuration
├── package.json          # Node.js dependencies
├── postcss.config.js     # PostCSS configuration
├── tailwind.config.ts    # Tailwind CSS configuration
└── tsconfig.json         # TypeScript configuration
```

## 🔌 API Endpoints

### Authentication
- `POST /api/users/signup` - Register a new user
- `POST /api/users/login` - User login
- `GET /api/users/logout` - User logout
- `GET /api/users/me` - Get current user details
- `POST /api/users/update` - Update user profile
- `POST /api/users/updatePassword` - Change password

### Projects
- `GET /api/users/projects` - Get all user projects
- `POST /api/users/project` - Create a new project
- `PUT /api/users/updateproject` - Update project details
- `DELETE /api/users/deleteproject` - Delete a project

### Machine Learning
- `POST /api/users/scripts` - Execute ML training scripts
- `GET /api/users/graphs` - Retrieve model graphs and visualizations

## 🎨 Features in Detail

### Creating a Project
1. Navigate to the home page after logging in
2. Click "New Project" button
3. Enter project name
4. Select an algorithm (Linear Regression, Logistic Regression, or KNN)
5. Click "Add Project"

### Training a Model
1. Open your created project
2. Upload a CSV file with your training data
3. Select features and target variable
4. Configure training parameters
5. Click "Train Model"
6. View training results and graphs

### Making Predictions
1. Navigate to trained model page
2. Enter feature values in the prediction form
3. Click "Predict"
4. View prediction results

## 🌙 Theme Support

NeuroFlow supports both light and dark themes:
- Toggle using the theme switcher in the navigation bar
- Theme preference is saved in local storage
- Automatically applies to all components

## 🔒 Security Features

- JWT-based authentication with secure tokens
- Password hashing using bcryptjs
- Protected API routes with middleware
- Input validation and sanitization
- Secure session management
- CORS configuration

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Vision Jain**
- GitHub: [@visionjain](https://github.com/visionjain)

## 🙏 Acknowledgments

- Next.js team for the amazing framework
- Vercel for deployment platform
- MongoDB for database services
- All open-source contributors

## 📞 Support

For support, please open an issue in the [GitHub repository](https://github.com/visionjain/NeuroFlow/issues).

## 🗺️ Roadmap

- [ ] Add more ML algorithms (SVM, Decision Trees, Random Forest)
- [ ] Implement model comparison features
- [ ] Add data visualization dashboard
- [ ] Export trained models
- [ ] Collaborative project sharing
- [ ] Real-time training progress
- [ ] Model performance metrics
- [ ] Automated hyperparameter tuning

---

Made with ❤️ by Vision Jain

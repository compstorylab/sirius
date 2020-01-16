const path = require('path');

module.exports = {
    entry:{
        graph: __dirname + '/client/ts/index.ts',
    },
    output:{
        path: __dirname + '/static/scripts',
        filename: "[name].js",
    },
    resolve:{
        extensions: [ '.tsx', '.ts', '.js' ],
    },
    module:{
        rules: [
          {
            test: /\.tsx?$/,
            use: 'ts-loader',
            exclude: /node_modules/,
          },
        ],

    },
    mode: 'development',
}
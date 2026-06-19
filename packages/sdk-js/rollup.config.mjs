import typescript from '@rollup/plugin-typescript';
import terser from '@rollup/plugin-terser';

export default [
  {
    input: 'src/index.ts',
    output: [
      { file: 'dist/litert-sdk.cjs.js', format: 'cjs', sourcemap: true },
      { file: 'dist/litert-sdk.esm.js', format: 'es', sourcemap: true },
      {
        file: 'dist/litert-sdk.umd.js',
        format: 'umd',
        name: 'LitertSDK',
        sourcemap: true,
        plugins: [terser()],
      },
      {
        file: 'dist/litert-sdk.min.js',
        format: 'iife',
        name: 'LitertSDK',
        plugins: [terser()],
      },
    ],
    plugins: [typescript({ tsconfig: './tsconfig.json' })],
  },
];

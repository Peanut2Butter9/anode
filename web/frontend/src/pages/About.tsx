export default function About() {
  return (
    <article className="mx-auto max-w-3xl space-y-6 text-sm leading-relaxed">
      <h1 className="text-2xl font-bold">About AnoDe</h1>
      <p>
        AnoDe started as a Machine Learning capstone project at the Department of Mathematics, Universitas Gadjah Mada.
        It was built by Albertus Reinaldo Manibuy, Daffa Elgo Santosa, Harun Yahya, Jeremy Christian Hutagalung and
        Moh Hasbi Alfarizi. This version is a from-scratch reproduction by Moh Hasbi Alfarizi. The benchmark was rerun
        with a stricter protocol, and the platform was rebuilt as this website.
      </p>
      <section className="card space-y-2 p-5">
        <h2 className="font-semibold">How the platform works</h2>
        <ol className="list-decimal space-y-1.5 pl-5 muted">
          <li>Each product has a model chosen by the benchmark, with a decision threshold set from normal images only (mean + 3σ).</li>
          <li>An inspection scores every unit. Scores are shown as a multiple of the threshold, with a heatmap where the model localises the defect.</li>
          <li>Finished batches go into a P-chart of the defect rate and a capability index (Cpk) of the anomaly score against the threshold as the upper spec limit.</li>
        </ol>
      </section>
      <section className="card space-y-2 p-5">
        <h2 className="font-semibold">Models</h2>
        <ul className="list-disc space-y-1.5 pl-5 muted">
          <li>PatchCore: Roth et al., <i>Towards Total Recall in Industrial Anomaly Detection</i>, CVPR 2022. Our implementation, deployed for the MVTec AD products and for custom products.</li>
          <li>CSAD: Hsieh &amp; Lai, <i>CSAD: Unsupervised Component Segmentation for Logical Anomaly Detection</i>, BMVC 2024. Released checkpoints, deployed for the MVTec LOCO AD products.</li>
          <li>GLASS: Chen et al., <i>A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization</i>, ECCV 2024. In the benchmark only (see the Benchmark page for why).</li>
        </ul>
      </section>
      <section className="card space-y-2 p-5">
        <h2 className="font-semibold">Data and license</h2>
        <p className="muted">
          Sample images and the training data of every model come from MVTec AD (Bergmann et al., IJCV 2021) and MVTec
          LOCO AD (Bergmann et al., IJCV 2022), © MVTec Software GmbH, licensed{' '}
          <a className="underline" href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" rel="noreferrer">CC BY-NC-SA 4.0</a>.
          This site is a non-commercial research demo.
        </p>
      </section>
      <section className="card space-y-2 p-5">
        <h2 className="font-semibold">Limitations</h2>
        <ul className="list-disc space-y-1.5 pl-5 muted">
          <li>Models are only as good as their normal images. A product photographed differently from its training set will be flagged.</li>
          <li>This public demo is a free static site: results for the sample images were computed in advance by the deployed models. Live inference (your own images, camera, custom products) runs when you start the AnoDe server locally: about 180 ms per image on a laptop CPU.</li>
          <li>Simulated history on the report pages is labelled as such. It is built from real test-set scores.</li>
        </ul>
      </section>
    </article>
  )
}

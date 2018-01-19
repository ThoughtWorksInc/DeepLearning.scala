parallelExecution in Global := false

includeFilter in unmanagedSources := (includeFilter in unmanagedSources).value && new SimpleFileFilter(_.isFile)

lazy val DeepLearning = project

lazy val `plugins-Layers` = project.dependsOn(DeepLearning, `plugins-Differentiables`)

lazy val `plugins-Weights` = project.dependsOn(DeepLearning, `plugins-Differentiables`)

lazy val `plugins-Names` = project.dependsOn(`plugins-Differentiables`)

lazy val `plugins-Logging` = project.dependsOn(`plugins-Differentiables`)

lazy val `plugins-Operators` = project

lazy val `plugins-Tensors` =
  project.dependsOn(ProjectRef(file("Expressions.scala"), "Expressions"), ProjectRef(file("Compute.scala"), "OpenCL"))

lazy val `plugins-FloatTraining` = project.dependsOn(`plugins-Training`)

lazy val `plugins-FloatLiterals` = project.dependsOn(`DeepLearning`)

lazy val `plugins-FloatWeights` = project.dependsOn(`plugins-Weights`)

lazy val `plugins-FloatLayers` =
  project.dependsOn(
    `plugins-Layers`,
    `plugins-Operators`,
    `plugins-FloatLiterals` % Test,
    `plugins-FloatTraining` % Test
  )

lazy val `plugins-CumulativeFloatLayers` =
  project.dependsOn(
    DeepLearning % "test->test",
    `plugins-FloatLayers`,
    `plugins-FloatTraining` % Test,
    `plugins-FloatLiterals` % Test,
    `plugins-FloatWeights` % Test
  )

lazy val `plugins-Training` = project.dependsOn(DeepLearning)

lazy val FloatRegex = """(?i:float)""".r

def copyAndReplace(floatProject: Project) = Def.task {
  for {
    floatFile <- (unmanagedSources in Compile in floatProject).value
    floatDirectory <- (unmanagedSourceDirectories in Compile in floatProject).value
    relativeFile <- floatFile.relativeTo(floatDirectory)
  } yield {
    val floatSource = IO.read(floatFile, scala.io.Codec.UTF8.charSet)
    val doubleSource = FloatRegex.replaceAllIn(floatSource, { m =>
      m.matched match {
        case "Float" => "Double"
        case "float" => "double"
      }
    })
    val outputFile = (sourceManaged in Compile).value / relativeFile.getPath
    IO.write(outputFile, doubleSource, scala.io.Codec.UTF8.charSet)
    outputFile
  }
}

lazy val `plugins-DoubleWeights` =
  project
    .dependsOn(`plugins-Weights`)
    .settings(sourceGenerators in Compile += copyAndReplace(`plugins-FloatWeights`).taskValue)

lazy val `plugins-DoubleTraining` =
  project
    .dependsOn(`plugins-Training`)
    .settings(sourceGenerators in Compile += copyAndReplace(`plugins-FloatTraining`).taskValue)

lazy val `plugins-DoubleLiterals` =
  project
    .dependsOn(`DeepLearning`)
    .settings(sourceGenerators in Compile += copyAndReplace(`plugins-FloatLiterals`).taskValue)

lazy val `plugins-DoubleLayers` =
  project
    .dependsOn(`plugins-Layers`, `plugins-Operators`)
    .settings(sourceGenerators in Compile += copyAndReplace(`plugins-FloatLayers`).taskValue)

lazy val `plugins-CumulativeDoubleLayers` =
  project
    .dependsOn(`plugins-DoubleLayers`, `plugins-Operators`)
    .settings(sourceGenerators in Compile += copyAndReplace(`plugins-CumulativeFloatLayers`).taskValue)

lazy val `plugins-Builtins` =
  project.dependsOn(
    `plugins-Layers`,
    `plugins-Weights`,
    `plugins-Logging`,
    `plugins-Names`,
    `plugins-Operators`,
    `plugins-FloatTraining`,
    `plugins-FloatLiterals`,
    `plugins-FloatWeights`,
    `plugins-FloatLayers`,
    `plugins-CumulativeFloatLayers`,
    `plugins-DoubleTraining`,
    `plugins-DoubleLiterals`,
    `plugins-DoubleWeights`,
    `plugins-DoubleLayers`,
    `plugins-CumulativeDoubleLayers`,
    DeepLearning % "test->test"
  )
lazy val `plugins-Differentiables` = project

lazy val `plugins-OpenCLBuffers` =
  project.dependsOn(
    DeepLearning,
    `plugins-FloatTraining` % Test,
    `plugins-Weights`,
    `plugins-Layers`,
    `plugins-FloatLayers`,
    `plugins-Logging`,
    `plugins-Names`,
    ProjectRef(file("Expressions.scala"), "Expressions"),
    ProjectRef(file("Compute.scala"), "OpenCLCodeGenerator"),
    ProjectRef(file("Compute.scala"), "OpenCL")
  )

publishArtifact := false

lazy val unidoc =
  project
    .enablePlugins(StandaloneUnidoc, TravisUnidocTitle)
    .settings(
      unidocProjectFilter in ScalaUnidoc in BaseUnidocPlugin.autoImport.unidoc := inAggregates(LocalRootProject),
      addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.3"),
      addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
      scalacOptions += "-Xexperimental",
      scalacOptions += "-Ypartial-unification"
    )

organization in ThisBuild := "com.thoughtworks.deeplearning"

crossScalaVersions := Seq("2.12.4")

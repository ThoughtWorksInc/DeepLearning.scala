parallelExecution in Global := false

// TODO: Rename to OpenCLLayer
lazy val DifferentiableKernel =
  project.dependsOn(
    OpenCL,
    OpenCLCodeGenerator,
    TapeTaskFactory,
    FutureIsomorphism,
    AsynchronousSemaphore
  )

lazy val OpenCLCodeGenerator = project.dependsOn(Memory)

//lazy val CheckedTape = project.dependsOn(Layer, Closeables)

// TODO: Move to a separate repository
lazy val Memory = project

lazy val Tape =
  project.dependsOn(ProjectRef(file("RAII.scala"), "packageJVM"), LogRecords, ProjectRef(file("RAII.scala"), "Do"))

lazy val TapeTaskFactory = project.dependsOn(Tape, ProjectRef(file("RAII.scala"), "Do"), Caller)

lazy val Closeables = project

lazy val Caller = project

includeFilter in unmanagedSources := (includeFilter in unmanagedSources).value && new SimpleFileFilter(_.isFile)

lazy val OpenCL = project.dependsOn(Closeables, Memory, ProjectRef(file("RAII.scala"), "ResourceFactoryTJVM"))

//lazy val LayerFactory = project.dependsOn(DifferentiableKernel)

lazy val `differentiable-float` = project.dependsOn(TapeTask, TapeTaskFactory, PolyFunctions, Caller)

lazy val `differentiable-double` = project.dependsOn(TapeTask, TapeTaskFactory, PolyFunctions, Caller)

val FloatRegex = """(?i:float)""".r

sourceGenerators in Compile in `differentiable-double` += Def.task {
  for {
    floatFile <- (unmanagedSources in Compile in `differentiable-float`).value
    relativeFile <- floatFile.relativeTo((sourceDirectory in Compile in `differentiable-float`).value)
  } yield {
    val floatSource = IO.read(floatFile, scala.io.Codec.UTF8.charSet)

    val doubleSource = FloatRegex.replaceAllIn(floatSource, { m =>
      m.matched match {
        case "Float" => "Double"
        case "float" => "double"
      }
    })

    val outputFile = (sourceManaged in Compile in `differentiable-double`).value / relativeFile.getPath
    IO.write(outputFile, doubleSource, scala.io.Codec.UTF8.charSet)
    outputFile
  }
}.taskValue

lazy val ToTapeTask = project.dependsOn(Tape, ProjectRef(file("RAII.scala"), "Do"))

lazy val PolyFunctions = project.dependsOn(ToTapeTask)

lazy val TapeTask = project.dependsOn(Tape, ProjectRef(file("RAII.scala"), "Do"))

lazy val LogRecords = project.dependsOn(Caller)

lazy val AsynchronousSemaphore = project

lazy val FutureIsomorphism = project

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(StandaloneUnidoc, TravisUnidocTitle)
  .settings(
    UnidocKeys.unidocProjectFilter in ScalaUnidoc in UnidocKeys.unidoc := inAggregates(LocalRootProject),
    addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.3"),
    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
    scalacOptions += "-Xexperimental"
  )

organization in ThisBuild := "com.thoughtworks.deeplearning"

crossScalaVersions := Seq("2.11.11", "2.12.2")

fork in ThisBuild in Test := true

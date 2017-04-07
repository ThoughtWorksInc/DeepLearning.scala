parallelExecution in Global := false
//
//lazy val DifferentiableKernel =
//  project.dependsOn(
//    OpenCL,
//    OpenCLCodeGenerator,
//    Layer,
//    ProjectRef(file("Future.scala"), "concurrent-Execution"),
//    ProjectRef(file("Future.scala"), "sde-task"),
//    ProjectRef(file("Future.scala"), "concurrent-Converters") % Test,
//    Symbolic % Test
//  )

//lazy val OpenCLCodeGenerator = project.dependsOn(Memory)

//lazy val Layer = project.dependsOn(ProjectRef(file("RAII.scala"), "ResourceFactoryTJVM"))

//lazy val CumulativeTape = project.dependsOn(Layer, ProjectRef(file("RAII.scala"), "Shared"))

//lazy val CheckedTape = project.dependsOn(Layer, Closeables)

lazy val Memory = project

lazy val Tape = project

//lazy val Closeables = project.dependsOn(ProjectRef(file("Future.scala"), "sde-task"),
//                                        ProjectRef(file("Future.scala"), "concurrent-Converters") % Test)

//// TODO: Rename to ToLiteral?
//lazy val Symbolic = project.dependsOn(Layer)

includeFilter in unmanagedSources := (includeFilter in unmanagedSources).value && new SimpleFileFilter(_.isFile)

//lazy val OpenCL = project.dependsOn(ProjectRef(file("RAII.scala"), "ResourceFactoryTJVM"), Memory)

//lazy val LayerFactory = project.dependsOn(DifferentiableKernel)

//lazy val DifferentiableFloat =
//  project.dependsOn(Layer,
//                    CumulativeTape,
//                    Symbolic % Test)

//lazy val DifferentiableDouble =
//  project.dependsOn(Layer,
//                    CumulativeTape,
//                    CheckedTape,
//                    ProjectRef(file("Future.scala"), "concurrent-Converters") % Test,
//                    Symbolic % Test)
//
//lazy val DifferentiableInt =
//  project.dependsOn(Layer,
//                    CumulativeTape,
//                    CheckedTape,
//                    DifferentiableFloat,
//                    ProjectRef(file("Future.scala"), "sde-task") % Test,
//                    Symbolic % Test)
//
//val FloatRegex = """(?i:float)""".r
//
//sourceGenerators in Compile in DifferentiableDouble += Def.task {
//  for {
//    floatFile <- (unmanagedSources in Compile in DifferentiableFloat).value
//    relativeFile <- floatFile.relativeTo((sourceDirectory in Compile in DifferentiableFloat).value)
//  } yield {
//    val floatSource = IO.read(floatFile, scala.io.Codec.UTF8.charSet)
//
//    val doubleSource = FloatRegex.replaceAllIn(floatSource, { m =>
//      m.matched match {
//        case "Float" => "Double"
//        case "float" => "double"
//      }
//    })
//
//    val outputFile = (sourceManaged in Compile in DifferentiableDouble).value / relativeFile.getPath
//    IO.write(outputFile, doubleSource, scala.io.Codec.UTF8.charSet)
//    outputFile
//  }
//}.taskValue

crossScalaVersions := Seq("2.11.8", "2.12.1")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(
    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
    scalaOrganization in updateSbtClassifiers := (scalaOrganization in Global).value,
    scalaOrganization := "org.typelevel",
    scalacOptions += "-Yliteral-types"
  )

organization in ThisBuild := "com.thoughtworks.deeplearning"

fork in Test := true
